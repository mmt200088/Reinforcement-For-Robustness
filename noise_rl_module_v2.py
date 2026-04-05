"""noise_rl_module_v2.py — 重构版噪声 RL 模块（V2 sparse-reward + noise-debt + MC terminal）

核心变更（相对 noise_rl_module.py）：
  1. 中间步不做 dense probe / completion 评估，reward = 0（纯稀疏奖励）。
  2. 终止步使用 MC=N 次 repeated evaluation，带方差惩罚。
  3. 连续特征维度 5 维，新增 noise_debt_ratio（CKKS 方差债务的 log 归一化）。
  4. 网络移除 mean_aux_head（单 critic），buffer 移除 mean_perf_targets。
  5. 确认失败直接写回 -50 惩罚到 buffer。
  6. PPO 更新中仅有单个 value head loss。
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from function_handler import (
    INPUT_NOISE_ALLOWED_SCALING_FACTORS,
    WEIGHT_NOISE_ALLOWED_SCALING_FACTORS,
    WFFN1_NOISE_ALLOWED_SCALING_FACTORS,
)


# ===========================================================================
# 超参数（Hyperparameters）
# ===========================================================================

# GTrXL 网络结构
NOISE_STAGE_GTRXL_D_MODEL = 256
NOISE_STAGE_GTRXL_N_HEADS = 8
NOISE_STAGE_GTRXL_N_LAYERS = 4
NOISE_STAGE_GTRXL_D_FF = 512
NOISE_STAGE_GTRXL_DROPOUT = 0.1

# PPO 训练
NOISE_STAGE_PPO_MAX_EPISODES = 40000
NOISE_STAGE_PPO_EPS_CLIP = 0.12
NOISE_STAGE_PPO_K_EPOCHS = 6
# 定长稀疏回报 episode：gamma=1 使终局奖励/惩罚对每一步 GAE 同等负责（避免 0.99^11 稀释）。
NOISE_STAGE_PPO_GAMMA = 1.0
# 终局含大幅惩罚时放宽 value clip，减轻 Critic 被 Huber+clip 拖死的问题。
NOISE_STAGE_VALUE_CLIP_RANGE = 1.0
NOISE_STAGE_GTRXL_WARMUP_MODE = "constant"
NOISE_STAGE_GTRXL_WARMUP_UPDATES = 0
NOISE_STAGE_GTRXL_SHORT_WARMUP_UPDATES = 20

# 分段评测段数（所有评测环节统一使用分段法；实际段数受 MIN_SAMPLES_PER_SEGMENT 自动下调保护）
NOISE_STAGE_BASELINE_SEGMENTS = 5         # baseline / worst 参考曲线
NOISE_STAGE_BEST_TEST_SEGMENTS = 5        # 初始守擂（initial incumbent）
NOISE_STAGE_MC_TRAIN_SAMPLES = 5          # 训练期终止步 MC 分段数
NOISE_STAGE_MC_CONFIRM_SEGMENTS = 5       # 挑战者确认（challenger confirmation）
NOISE_STAGE_MIN_SAMPLES_PER_SEGMENT = 80  # 每段最少样本数，段数超限时自动缩减

# 方差惩罚（Stability Penalty）
NOISE_STAGE_STABILITY_LAMBDA = 5.0        # lambda_stab 方差惩罚权重
NOISE_STAGE_STABILITY_BASE_STD_TOLERANCE = 1.2  # 基线标准差容忍倍数

# 确认失败惩罚
NOISE_STAGE_CONFIRM_FAIL_PENALTY = -50.0  # 确认失败写回 buffer 的惩罚

# 兼容旧 checkpoint / 元数据；连续特征中的 noise_debt_ratio 已改用 log2 映射（见 get_continuous_features）。
NOISE_STAGE_NOISE_DEBT_LOG_ALPHA = 1.0

# reward 权重
NOISE_STAGE_TAIL_MARGIN_WEIGHT = 1.00
NOISE_STAGE_TAIL_VIOLATION_WEIGHT = 1.0   # 从 1.25 降低，鼓励探索
NOISE_STAGE_COST_WEIGHT = 0.25            # 从 0.05 提升，鼓励降成本

# 性能指标权重
NOISE_STAGE_PERF_WEIGHT_LOSS = 0.15
NOISE_STAGE_PERF_WEIGHT_M1 = 0.425
NOISE_STAGE_PERF_WEIGHT_M2 = 0.425
NOISE_STAGE_BARRIER_WEIGHT_LOSS = 0.10
NOISE_STAGE_BARRIER_WEIGHT_M1 = 0.45
NOISE_STAGE_BARRIER_WEIGHT_M2 = 0.45

# 动态约束
NOISE_STAGE_DYNAMIC_LIMIT_QUARTILE = 0.2

# 挑战者触发
NOISE_STAGE_BEST_TEST_TRIGGER_MARGIN = 0.0
NOISE_STAGE_BEST_TEST_SAFE_RATE_MIN = 0.93
NOISE_STAGE_BEST_TEST_CVAR_MAX = 0.05
NOISE_STAGE_BEST_TEST_MAX_ENTRY_SCORE_DROP = 0.015
NOISE_STAGE_BEST_TEST_ENTRY_STD_MULTIPLIER = 2.0

# V2 连续特征维度 (cost_deviation, complexity_debt, progress, noise_debt_ratio, remaining_budget_ratio)
NOISE_STAGE_V2_CONT_DIM = 5

# 进度保存
NOISE_STAGE_PROGRESS_SAVE_INTERVAL = 10000
NOISE_STAGE_PROGRESS_DIR = os.path.join("rl_results", "noise_rl_progress")

# 状态常量
NOISE_STAGE_WEIGHT_TOL = 1e-6
NOISE_STAGE_STATUS_OK = "ok"
NOISE_STAGE_STATUS_NO_STABLE_FEASIBLE = "no_stable_feasible_candidate"

# reward 裁剪
NOISE_STAGE_REWARD_CLIP_MIN = -5.0
NOISE_STAGE_REWARD_CLIP_MAX = 5.0

# checkpoint 文件名
NOISE_STAGE_CHECKPOINT_FILENAME = "noise_rl_checkpoint.pt"
STAGE1_CHECKPOINT_FILENAME = "stage1_rl_checkpoint.pt"


# ===========================================================================
# Helper functions（辅助函数）
# ===========================================================================

def _get_low_risk_noise_configuration(evaluator):
    """构建低风险参考噪声配置：每类噪声取允许集合中的最大 scaling factor。"""
    total_layers = evaluator.total_layers
    max_input_sf = max(INPUT_NOISE_ALLOWED_SCALING_FACTORS)
    max_weight_sf = max(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS)
    max_wffn1_sf = max(WFFN1_NOISE_ALLOWED_SCALING_FACTORS)
    return {
        "input_noise_scaling_factors": np.full(total_layers, max_input_sf, dtype=int),
        "wq_noise_scaling_factors": np.full(total_layers, max_weight_sf, dtype=int),
        "wk_noise_scaling_factors": np.full(total_layers, max_weight_sf, dtype=int),
        "wv_noise_scaling_factors": np.full(total_layers, max_weight_sf, dtype=int),
        "wo_noise_scaling_factors": np.full(total_layers, max_weight_sf, dtype=int),
        "wffn1_noise_scaling_factors": np.full(total_layers, max_wffn1_sf, dtype=int),
        "wffn2_noise_scaling_factors": np.full(total_layers, max_weight_sf, dtype=int),
    }


def _get_worst_case_noise_configuration(evaluator):
    """构建最差噪声配置：每类噪声取允许集合中的最小 scaling factor（噪声最大）。"""
    total_layers = evaluator.total_layers
    min_input_sf = min(INPUT_NOISE_ALLOWED_SCALING_FACTORS)
    min_weight_sf = min(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS)
    min_wffn1_sf = min(WFFN1_NOISE_ALLOWED_SCALING_FACTORS)
    return {
        "input_noise_scaling_factors": np.full(total_layers, min_input_sf, dtype=int),
        "wq_noise_scaling_factors": np.full(total_layers, min_weight_sf, dtype=int),
        "wk_noise_scaling_factors": np.full(total_layers, min_weight_sf, dtype=int),
        "wv_noise_scaling_factors": np.full(total_layers, min_weight_sf, dtype=int),
        "wo_noise_scaling_factors": np.full(total_layers, min_weight_sf, dtype=int),
        "wffn1_noise_scaling_factors": np.full(total_layers, min_wffn1_sf, dtype=int),
        "wffn2_noise_scaling_factors": np.full(total_layers, min_weight_sf, dtype=int),
    }


def _compute_dynamic_limits(
    base_loss, base_p, base_s,
    worst_loss, worst_p, worst_s,
    quartile=NOISE_STAGE_DYNAMIC_LIMIT_QUARTILE,
):
    """根据 baseline 和 worst-case 指标动态计算约束上下限。

    Loss 越小越好；metric1/metric2 越大越好。
    特殊情况（统计噪声导致插值方向反常）时不插值，使用 baseline 本身。
    """
    if base_loss > worst_loss:
        limit_loss = float(base_loss)
    else:
        limit_loss = float(base_loss + quartile * (worst_loss - base_loss))

    if base_p < worst_p:
        limit_p = float(base_p)
    else:
        limit_p = float(base_p + quartile * (worst_p - base_p))

    if base_s < worst_s:
        limit_s = float(base_s)
    else:
        limit_s = float(base_s + quartile * (worst_s - base_s))

    return {
        "loss": limit_loss,
        "metric1": limit_p,
        "metric2": limit_s,
    }


def _compute_dynamic_std_upper_bound(
    baseline_std, worst_std, quartile=NOISE_STAGE_DYNAMIC_LIMIT_QUARTILE,
):
    """与 `_compute_dynamic_limits` 中 loss 维同构：std 越大越不利。

    - 正常：上界 = baseline_std + quartile * (worst_std - baseline_std)
    - 若 worst_std < baseline_std（最坏情形反而更稳），不外推放宽，取 baseline_std。
    """
    if baseline_std > worst_std:
        return float(baseline_std)
    return float(baseline_std + quartile * (worst_std - baseline_std))


def _auto_adjust_segments(dataset_size, requested_segments, log_fn=None,
                          min_samples=NOISE_STAGE_MIN_SAMPLES_PER_SEGMENT):
    """根据数据集大小自动缩减分段数，保证每段样本数 >= min_samples。

    返回实际使用的段数（>= 1）。
    """
    requested_segments = max(1, int(requested_segments))
    if dataset_size <= 0:
        return 1
    max_n = max(1, dataset_size // min_samples)
    if requested_segments <= max_n:
        return requested_segments
    adjusted = max_n
    if log_fn is not None:
        log_fn(
            f"  [自动调整] 指定分段数 N={requested_segments} 导致每段仅 "
            f"{dataset_size // requested_segments} 个样本（< {min_samples}），"
            f"已自动缩减为 N={adjusted}（每段约 {dataset_size // adjusted} 个样本）"
        )
    return adjusted


def _log_rounded_box(log_fn, lines, indent="  ", min_inner_width=8):
    """圆角框日志。"""
    stripped = [str(x) for x in lines]
    w = max((len(s) for s in stripped), default=0)
    w = max(w, min_inner_width)
    bar = "\u2500" * (w + 4)
    log_fn(f"{indent}\u256d{bar}\u256e")
    for s in stripped:
        log_fn(f"{indent}\u2502 {s.ljust(w)} \u2502")
    log_fn(f"{indent}\u2570{bar}\u256f")


def _write_rounded_box_to_file(f, lines, indent=""):
    """与 _log_rounded_box 相同规则，写入文本文件。"""
    stripped = [str(x) for x in lines]
    w = max((len(s) for s in stripped), default=0)
    w = max(w, 8)
    bar = "\u2500" * (w + 4)
    f.write(f"{indent}\u256d{bar}\u256e\n")
    for s in stripped:
        f.write(f"{indent}\u2502 {s.ljust(w)} \u2502\n")
    f.write(f"{indent}\u2570{bar}\u256f\n")


def _noise_log_major_rule(log_fn, title, width=68):
    """二阶段噪声日志：小节顶栏。"""
    bar = "\u2550" * width
    log_fn("")
    log_fn(bar)
    log_fn(title if title.startswith(" ") else f"  {title}")
    log_fn(bar)


def _noise_block_title(log_fn, title):
    """块标题，便于扫读。"""
    log_fn("")
    log_fn(f"  \u3010{title}\u3011")


def _validate_stage2_alpha_weights(alpha_perf, alpha_cost, tol=NOISE_STAGE_WEIGHT_TOL):
    alpha_perf = float(alpha_perf)
    alpha_cost = float(alpha_cost)
    if not np.isclose(alpha_perf + alpha_cost, 1.0, atol=tol):
        raise ValueError(
            f"Stage-2 final reward alpha weights must sum to 1.0: "
            f"perf={alpha_perf}, cost={alpha_cost}"
        )
    return alpha_perf, alpha_cost


def _resolve_stage2_metric_weights(num_metrics, weight_map, label):
    active_keys = ["loss", "metric1"]
    if num_metrics > 1:
        active_keys.append("metric2")
        total = sum(float(weight_map[key]) for key in active_keys)
        if not np.isclose(total, 1.0, atol=NOISE_STAGE_WEIGHT_TOL):
            raise ValueError(
                f"{label} must sum to 1.0 for multi-metric tasks, got {total:.6f}"
            )
        return {key: float(weight_map[key]) for key in active_keys}

    active_total = sum(float(weight_map[key]) for key in active_keys)
    if active_total <= 0:
        raise ValueError(f"{label} active weights must sum to a positive value")
    return {
        key: float(weight_map[key]) / active_total
        for key in active_keys
    }


def _warmup_status_zh(status):
    m = {"constant": "\u6052\u5b9a\uff08constant\uff09",
         "warmup": "\u9884\u70ed\uff08warmup\uff09",
         "normal": "\u5e38\u89c4\uff08normal\uff09"}
    return m.get(status, status)


def _noise_stage_status_zh(status, ok_token, no_stable_token):
    m = {
        ok_token: "\u6b63\u5e38\uff08ok\uff09",
        no_stable_token: "\u65e0\u7a33\u5b9a\u53ef\u884c\u89e3\uff08no_stable_feasible_candidate\uff09",
    }
    return m.get(status, status)


def _compute_ckks_variance(exponent):
    """CKKS 物理：方差 ~ 1/Delta^2, Delta = 2^exp, 故 variance = 2^(-2*exp)。"""
    return 2.0 ** (-2.0 * float(exponent))


def _compute_max_single_layer_debt():
    """假设每个 noise type 取最小 scaling factor（噪声最大），计算单层最大方差债务。"""
    min_input_exp = min(INPUT_NOISE_ALLOWED_SCALING_FACTORS)
    min_weight_exp = min(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS)
    min_wffn1_exp = min(WFFN1_NOISE_ALLOWED_SCALING_FACTORS)
    return (
        _compute_ckks_variance(min_input_exp)        # input
        + _compute_ckks_variance(min_weight_exp)     # wq
        + _compute_ckks_variance(min_weight_exp)     # wk
        + _compute_ckks_variance(min_weight_exp)     # wv
        + _compute_ckks_variance(min_weight_exp)     # wo
        + _compute_ckks_variance(min_wffn1_exp)      # wffn1
        + _compute_ckks_variance(min_weight_exp)     # wffn2
    )


def _compute_min_single_layer_debt():
    """每层若全部取最大 scaling factor（噪声最小），单层 CKKS 方差债务下界。"""
    max_input_exp = max(INPUT_NOISE_ALLOWED_SCALING_FACTORS)
    max_weight_exp = max(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS)
    max_wffn1_exp = max(WFFN1_NOISE_ALLOWED_SCALING_FACTORS)
    vw = _compute_ckks_variance(max_weight_exp)
    return (
        _compute_ckks_variance(max_input_exp)
        + vw + vw + vw + vw
        + _compute_ckks_variance(max_wffn1_exp)
        + vw
    )


def _write_noise_step_info(step_info, f):
    """简化版逐步诊断写入。"""
    f.write(f"  \u5168\u5c40\u6b65\u6570\uff08step_global\uff09: {step_info['step_global']}\n")
    f.write(f"  \u56de\u5408\u7f16\u53f7\uff08episode_id\uff09: {step_info['episode_id']}\n")
    f.write(f"  \u5c42\u7d22\u5f15\uff08layer_index\uff09: {step_info['layer_index']}\n")
    for key in ("curr_input_noise_scaling_factor", "curr_wq_noise_scaling_factor",
                "curr_wk_noise_scaling_factor", "curr_wv_noise_scaling_factor",
                "curr_wo_noise_scaling_factor", "curr_wffn1_noise_scaling_factor",
                "curr_wffn2_noise_scaling_factor"):
        if key in step_info:
            f.write(f"  {key}: {step_info[key]}\n")
    for key in ("critic_value", "accumulated_cost", "step_reward",
                "raw_final_reward", "final_selection_score",
                "noise_debt_ratio", "accumulated_noise_variance"):
        if step_info.get(key) is not None:
            f.write(f"  {key}: {step_info[key]}\n")
    for key in ("x_prob_dist", "wq_prob_dist", "wk_prob_dist", "wv_prob_dist",
                "wo_prob_dist", "wffn1_prob_dist", "wffn2_prob_dist"):
        if key in step_info:
            f.write(f"  {key}: {step_info[key]}\n")
    if step_info.get("mc_loss_mean") is not None:
        f.write(f"  MC: loss={step_info['mc_loss_mean']:.4f}+/-{step_info.get('mc_loss_std', 0):.4f}, "
                f"m1={step_info.get('mc_metric1_mean', 0):.4f}+/-{step_info.get('mc_metric1_std', 0):.4f}\n")


def _write_warning_report(warning_file, warnings, stage_label=""):
    """将奖励骤降警告写入报告。"""
    import datetime
    dt_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    title = "\u5f3a\u5316\u5b66\u4e60\u5956\u52b1\u9a9f\u964d\u8b66\u544a\u62a5\u544a"
    with open(warning_file, "w", encoding="utf-8") as f:
        f.write(f"=== {title} ===\n")
        f.write(f"\u9636\u6bb5: {stage_label}\n")
        f.write(f"\u66f4\u65b0\u65f6\u95f4: {dt_str}\n")
        f.write(f"\u7d2f\u8ba1\u8b66\u544a: {len(warnings)} \u6761\n\n")
        for i, w in enumerate(warnings, 1):
            ep_start, ep_end = w["episode_range"]
            f.write(f"--- \u8b66\u544a #{i} ---\n")
            f.write(f"  \u7c7b\u578b: {w['type']}\n")
            f.write(f"  \u6da8\u5e45: {w['drop']:.4f} (\u4e0a\u6b21={w['prev_avg']:.4f}, \u672c\u6b21={w['curr_avg']:.4f})\n")
            f.write(f"  \u56de\u5408\u8303\u56f4: {ep_start}-{ep_end}\n")
            for df in w.get("detail_files", []):
                f.write(f"    -> details/{df}\n")
            f.write("\n")


# ===========================================================================
# Checkpoint save / load helpers（断点续训辅助函数）
# ===========================================================================

def _serialize_numpy(obj):
    """递归将 numpy 数组转为 list，方便 torch.save 序列化。"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _serialize_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [_serialize_numpy(item) for item in obj]
        return type(obj)(converted) if isinstance(obj, tuple) else converted
    return obj


def _deserialize_numpy_keys(obj, keys_to_convert):
    """将指定 key 下的 list 恢复为 numpy int 数组。"""
    if not isinstance(obj, dict):
        return obj
    for key in keys_to_convert:
        if key in obj and obj[key] is not None:
            if isinstance(obj[key], list):
                obj[key] = np.array(obj[key], dtype=int)
    return obj


def save_noise_rl_checkpoint(
    path,
    noise_net,
    optimizer,
    episode,
    noise_ppo_update_count,
    episode_returns,
    episode_raw_final_rewards,
    episode_final_selection_scores,
    episode_losses,
    episode_metric1s,
    episode_metric2s,
    episode_entropies,
    best_final_selection_score,
    best_cost,
    best_noise_config,
    incumbent_best_noise_config,
    incumbent_best_signature,
    incumbent_mean_score,
    incumbent_history,
    ev_runtime_state,
    noise_prev_avg_reward,
    noise_warnings,
):
    """保存 Stage-2 噪声 RL 的完整训练状态（checkpoint）。"""
    checkpoint = {
        "version": 2,
        "completed_episodes": episode + 1,
        "noise_net_state_dict": noise_net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "noise_ppo_update_count": noise_ppo_update_count,
        "episode_returns": list(episode_returns),
        "episode_raw_final_rewards": list(episode_raw_final_rewards),
        "episode_final_selection_scores": list(episode_final_selection_scores),
        "episode_losses": list(episode_losses),
        "episode_metric1s": list(episode_metric1s),
        "episode_metric2s": list(episode_metric2s),
        "episode_entropies": list(episode_entropies),
        "best_final_selection_score": best_final_selection_score,
        "best_cost": best_cost,
        "best_noise_config": _serialize_numpy(best_noise_config),
        "incumbent_best_noise_config": _serialize_numpy(incumbent_best_noise_config),
        "incumbent_best_signature": incumbent_best_signature,
        "incumbent_mean_score": incumbent_mean_score,
        "incumbent_history": _serialize_numpy(incumbent_history),
        "ev_runtime_state": ev_runtime_state,
        "noise_prev_avg_reward": noise_prev_avg_reward,
        "noise_warnings": _serialize_numpy(noise_warnings),
    }
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save(checkpoint, path)


def load_noise_rl_checkpoint(path, noise_net, optimizer, device="cuda"):
    """加载 Stage-2 噪声 RL checkpoint，恢复训练状态。返回 checkpoint dict。"""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    noise_net.load_state_dict(checkpoint["noise_net_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # 恢复 numpy 数组
    _np_keys = [
        "input_noise_scaling_factors", "wq_noise_scaling_factors",
        "wk_noise_scaling_factors", "wv_noise_scaling_factors",
        "wo_noise_scaling_factors", "wffn1_noise_scaling_factors",
        "wffn2_noise_scaling_factors",
    ]
    if checkpoint.get("best_noise_config") is not None:
        checkpoint["best_noise_config"] = _deserialize_numpy_keys(
            checkpoint["best_noise_config"], _np_keys,
        )
    if checkpoint.get("incumbent_best_noise_config") is not None:
        checkpoint["incumbent_best_noise_config"] = _deserialize_numpy_keys(
            checkpoint["incumbent_best_noise_config"], _np_keys,
        )
    return checkpoint


def save_stage1_rl_checkpoint(
    path,
    gtrxl_net,
    optimizer,
    episode,
    gtrxl_ppo_update_count,
    episode_rewards,
    episode_losses,
    episode_metric1s,
    episode_metric2s,
    episode_entropies,
    best_reward,
    best_cost,
    best_config,
    search_best_config,
    global_best_config,
    window_best_reward,
    window_best_cost,
    window_best_config,
    ev_runtime_state,
    stage1_prev_avg_reward,
    stage1_warnings,
):
    """保存 Stage-1 RL 的完整训练状态（checkpoint）。"""
    checkpoint = {
        "version": 1,
        "completed_episodes": episode + 1,
        "gtrxl_net_state_dict": gtrxl_net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "gtrxl_ppo_update_count": gtrxl_ppo_update_count,
        "episode_rewards": list(episode_rewards),
        "episode_losses": list(episode_losses),
        "episode_metric1s": list(episode_metric1s),
        "episode_metric2s": list(episode_metric2s),
        "episode_entropies": list(episode_entropies),
        "best_reward": best_reward,
        "best_cost": best_cost,
        "best_config": _serialize_numpy(best_config),
        "search_best_config": _serialize_numpy(search_best_config),
        "global_best_config": _serialize_numpy(global_best_config),
        "window_best_reward": window_best_reward,
        "window_best_cost": window_best_cost,
        "window_best_config": _serialize_numpy(window_best_config),
        "ev_runtime_state": ev_runtime_state,
        "stage1_prev_avg_reward": stage1_prev_avg_reward,
        "stage1_warnings": _serialize_numpy(stage1_warnings),
    }
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save(checkpoint, path)


def load_stage1_rl_checkpoint(path, gtrxl_net, optimizer, device="cuda"):
    """加载 Stage-1 RL checkpoint，恢复训练状态。返回 checkpoint dict。"""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    gtrxl_net.load_state_dict(checkpoint["gtrxl_net_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    _np_keys = ["gelu", "softmax"]
    for cfg_key in ("best_config", "search_best_config", "global_best_config", "window_best_config"):
        if checkpoint.get(cfg_key) is not None:
            checkpoint[cfg_key] = _deserialize_numpy_keys(
                checkpoint[cfg_key], _np_keys,
            )
    return checkpoint


# ===========================================================================
# _NoiseRecurrentRolloutBuffer
# ===========================================================================

class _NoiseRecurrentRolloutBuffer:
    """V2 Rollout buffer: \u5355\u5934 critic\uff0c\u65e0 mean_perf_targets\u3002"""

    def __init__(self):
        self.episodes = []
        self._current = None

    def start_episode(self):
        self._current = {
            "cont_features": [],
            "layer_indices": [],
            "prev_actions": [],
            "actions": [],
            "logprobs": [],
            "rewards": [],
            "values": [],
            "dones": [],
        }

    def add_step(self, cont_feat, layer_idx, prev_actions, actions, logprob,
                 reward, value, done):
        self._current["cont_features"].append(cont_feat)
        self._current["layer_indices"].append(layer_idx)
        self._current["prev_actions"].append(prev_actions)
        self._current["actions"].append(actions)
        self._current["logprobs"].append(logprob)
        self._current["rewards"].append(reward)
        self._current["values"].append(value)
        self._current["dones"].append(done)

    def end_episode(self):
        self.episodes.append(self._current)
        self._current = None

    def clear(self):
        self.episodes.clear()

    @property
    def num_episodes(self):
        return len(self.episodes)

    def get_batch(self, device):
        cont_features = torch.stack([
            torch.stack(ep["cont_features"]) for ep in self.episodes
        ]).to(device)

        layer_indices = torch.stack([
            torch.tensor(ep["layer_indices"], dtype=torch.long)
            for ep in self.episodes
        ]).to(device)

        prev_actions = torch.stack([
            torch.stack(ep["prev_actions"]) for ep in self.episodes
        ]).to(device)

        actions = torch.stack([
            torch.stack(ep["actions"]) for ep in self.episodes
        ]).to(device)

        logprobs = torch.stack([
            torch.stack(ep["logprobs"]) for ep in self.episodes
        ]).to(device)

        rewards = torch.tensor([
            ep["rewards"] for ep in self.episodes
        ], dtype=torch.float32).to(device)

        values = torch.stack([
            torch.stack(ep["values"]) for ep in self.episodes
        ]).to(device)

        dones = torch.tensor([
            ep["dones"] for ep in self.episodes
        ], dtype=torch.float32).to(device)

        return (cont_features, layer_indices, prev_actions, actions,
                logprobs, rewards, values, dones)


# ===========================================================================
# _NoiseGTrXLStrategyNetwork  (\u5355 critic head, \u65e0 mean_aux_head)
# ===========================================================================

class _NoiseGTrXLStrategyNetwork(nn.Module):
    """Second-stage GTrXL actor-critic with 7 independent noise-action heads.
    V2: \u5355\u5934 critic\uff0c\u65e0 mean_aux_head\u3002
    """

    action_names = ("x", "wq", "wk", "wv", "wo", "wffn1", "wffn2")

    def __init__(self, num_layers=12, d_model=64,
                 n_heads=4, n_gtrxl_layers=3,
                 d_ff=128, dropout=0.1,
                 gtrxl_block_cls=None,
                 lstm_pos_dim=16, lstm_proj_dim=32,
                 noise_stage_num_actions=7,
                 noise_stage_sos_tokens=None,
                 noise_stage_prev_action_embed_dim=4,
                 noise_stage_cont_dim=5,
                 noise_stage_action_dims=None):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self._action_dims = noise_stage_action_dims
        self._sos_tokens = noise_stage_sos_tokens

        self.embed_layer_idx = nn.Embedding(num_layers, lstm_pos_dim)
        self.prev_action_embeddings = nn.ModuleList([
            nn.Embedding(noise_stage_sos_tokens[i] + 1, noise_stage_prev_action_embed_dim)
            for i in range(noise_stage_num_actions)
        ])
        self.fc_continuous = nn.Sequential(
            nn.Linear(noise_stage_cont_dim, lstm_proj_dim),
            nn.LayerNorm(lstm_proj_dim),
            nn.SiLU()
        )

        token_input_dim = (
            lstm_pos_dim
            + noise_stage_num_actions * noise_stage_prev_action_embed_dim
            + lstm_proj_dim
        )
        self.input_proj = (
            nn.Identity() if token_input_dim == d_model
            else nn.Linear(token_input_dim, d_model)
        )

        self.gtrxl_blocks = nn.ModuleList([
            gtrxl_block_cls(d_model, n_heads, d_ff, dropout)
            for _ in range(n_gtrxl_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)

        self.actor_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.Tanh()
        )
        self.noise_heads = nn.ModuleDict({
            name: nn.Linear(64, noise_stage_action_dims[idx])
            for idx, name in enumerate(self.action_names)
        })

        # \u5355\u5934 critic
        self.critic_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self._causal_mask_cache = {}
        self._initialize_weights()

    def _initialize_weights(self):
        for module in [self.actor_head, self.critic_head, self.fc_continuous]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0.0)

        for head in self.noise_heads.values():
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.constant_(head.bias, 0.0)

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
        if seq_len not in self._causal_mask_cache or self._causal_mask_cache[seq_len].device != device:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
                diagonal=1,
            )
            self._causal_mask_cache[seq_len] = mask
        return self._causal_mask_cache[seq_len]

    def _build_tokens(self, cont_features, layer_indices, prev_actions):
        emb_l = self.embed_layer_idx(layer_indices)
        prev_embs = [
            emb(prev_actions[:, :, idx])
            for idx, emb in enumerate(self.prev_action_embeddings)
        ]
        feat_c = self.fc_continuous(cont_features)
        token_input = torch.cat([emb_l, *prev_embs, feat_c], dim=-1)
        return self.input_proj(token_input)

    def forward(self, cont_features, layer_indices, prev_actions, key_padding_mask=None):
        tokens = self._build_tokens(cont_features, layer_indices, prev_actions)
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

    def get_action_and_logprob(self, cont_features, layer_indices, prev_actions,
                               return_probs=False):
        logits_dict, values = self.forward(cont_features, layer_indices, prev_actions)
        value = values[:, -1].squeeze(0)

        actions = []
        probs = []
        logprob = torch.zeros((), dtype=torch.float32, device=cont_features.device)
        for name in self.action_names:
            logits = logits_dict[name][:, -1, :].squeeze(0)
            dist = Categorical(logits=logits)
            action = dist.sample()
            actions.append(action)
            logprob = logprob + dist.log_prob(action)
            if return_probs:
                probs.append(torch.softmax(logits, dim=-1))

        actions_tensor = torch.stack(actions)
        if return_probs:
            return actions_tensor, logprob, value, probs
        return actions_tensor, logprob, value

    def evaluate_actions(self, cont_features, layer_indices, prev_actions, actions):
        logits_dict, values = self.forward(cont_features, layer_indices, prev_actions)
        logprobs = torch.zeros_like(values)
        entropy = torch.zeros_like(values)
        for idx, name in enumerate(self.action_names):
            dist = Categorical(logits=logits_dict[name])
            logprobs = logprobs + dist.log_prob(actions[:, :, idx])
            entropy = entropy + dist.entropy()
        return logprobs, entropy, values


# ===========================================================================
# _NoiseOptEnv  (V2: \u7a00\u758f\u5956\u52b1 + \u566a\u58f0\u65b9\u5dee\u503a\u52a1 + MC \u7ec8\u7aef\u5956\u52b1)
# ===========================================================================

class _NoiseOptEnv:
    """V2 \u5668\uff1a\u4e2d\u95f4\u6b65 reward=0\uff0c\u7ec8\u6b62\u6b65 MC \u8bc4\u4f30 + \u65b9\u5dee\u60e9\u7f5a\u3002"""

    def __init__(self, total_layers, baseline_cost, baseline_metrics, evaluator,
                 fixed_gelu, fixed_softmax, constraint_limits=None, num_metrics=2,
                 input_noise_allowed=None, weight_noise_allowed=None, wffn1_noise_allowed=None,
                 input_noise_cost_map=None, weight_noise_cost_map=None, wffn1_noise_cost_map=None,
                 input_noise_scaling_map=None,
                 wq_noise_scaling_map=None, wk_noise_scaling_map=None,
                 wv_noise_scaling_map=None, wo_noise_scaling_map=None,
                 wffn1_noise_scaling_map=None, wffn2_noise_scaling_map=None,
                 input_noise_scaling_to_norm=None, weight_noise_scaling_to_norm=None,
                 wffn1_noise_scaling_to_norm=None,
                 noise_stage_sos_tokens=None, noise_stage_num_actions=7,
                 history_mask_value=0.0,
                 reward_clip_min=-5.0, reward_clip_max=5.0,
                 perf_weight_loss=0.15, perf_weight_m1=0.425, perf_weight_m2=0.425,
                 barrier_weight_loss=0.10, barrier_weight_m1=0.45, barrier_weight_m2=0.45,
                 mc_train_samples=5,
                 stability_lambda=5.0,
                 stability_base_std_tolerance=1.2,
                 baseline_loss_std=0.0,
                 baseline_m1_std=0.0,
                 dynamic_loss_std_cap=0.0,
                 dynamic_m1_std_cap=0.0,
                 dynamic_m2_std_cap=0.0,
                 noise_debt_log_alpha=1.0):
        self.total_layers = total_layers
        self.baseline_cost = baseline_cost
        self.baseline_loss, self.baseline_p, self.baseline_s = baseline_metrics
        self.evaluator = evaluator
        self.fixed_gelu = np.asarray(fixed_gelu, dtype=int)
        self.fixed_softmax = np.asarray(fixed_softmax, dtype=int)
        self.num_metrics = num_metrics

        self._input_noise_allowed = input_noise_allowed
        self._weight_noise_allowed = weight_noise_allowed
        self._wffn1_noise_allowed = (
            wffn1_noise_allowed if wffn1_noise_allowed is not None
            else weight_noise_allowed
        )
        self._input_noise_cost_map = input_noise_cost_map
        self._weight_noise_cost_map = weight_noise_cost_map
        self._wffn1_noise_cost_map = (
            wffn1_noise_cost_map if wffn1_noise_cost_map is not None
            else weight_noise_cost_map
        )
        self._input_noise_scaling_map = input_noise_scaling_map
        self._wq_map = wq_noise_scaling_map
        self._wk_map = wk_noise_scaling_map
        self._wv_map = wv_noise_scaling_map
        self._wo_map = wo_noise_scaling_map
        self._wffn1_map = wffn1_noise_scaling_map
        self._wffn2_map = wffn2_noise_scaling_map
        self._input_noise_to_norm = input_noise_scaling_to_norm
        self._weight_noise_to_norm = weight_noise_scaling_to_norm
        self._wffn1_noise_to_norm = (
            wffn1_noise_scaling_to_norm if wffn1_noise_scaling_to_norm is not None
            else weight_noise_scaling_to_norm
        )
        self._sos_tokens = noise_stage_sos_tokens
        self._num_actions = noise_stage_num_actions
        self._history_mask = history_mask_value
        self._reward_clip_min = reward_clip_min
        self._reward_clip_max = reward_clip_max

        self._perf_weights = _resolve_stage2_metric_weights(
            num_metrics,
            {"loss": perf_weight_loss, "metric1": perf_weight_m1, "metric2": perf_weight_m2},
            label="perf_weights",
        )
        self._barrier_weights = _resolve_stage2_metric_weights(
            num_metrics,
            {"loss": barrier_weight_loss, "metric1": barrier_weight_m1, "metric2": barrier_weight_m2},
            label="barrier_weights",
        )

        self._mc_train_samples = max(1, int(mc_train_samples))
        self._stability_lambda = float(stability_lambda)
        self._stability_base_std_tolerance = float(stability_base_std_tolerance)
        self._baseline_loss_std = float(baseline_loss_std)
        self._baseline_m1_std = float(baseline_m1_std)
        self._dynamic_loss_std_cap = float(dynamic_loss_std_cap)
        self._dynamic_m1_std_cap = float(dynamic_m1_std_cap)
        self._dynamic_m2_std_cap = float(dynamic_m2_std_cap)
        self._noise_debt_log_alpha = float(noise_debt_log_alpha)
        self._episode_progress = 0.0

        if constraint_limits is None:
            self.constraint_limits = {
                "loss": self.baseline_loss * 1.01,
                "metric1": self.baseline_p * 0.99,
                "metric2": self.baseline_s * 0.99,
            }
        else:
            self.constraint_limits = constraint_limits

        # \u6210\u672c\u8fb9\u754c
        min_input_cost = self._input_noise_cost_map[min(self._input_noise_allowed)]
        max_input_cost = self._input_noise_cost_map[max(self._input_noise_allowed)]
        min_generic_weight_cost = self._weight_noise_cost_map[min(self._weight_noise_allowed)]
        max_generic_weight_cost = self._weight_noise_cost_map[max(self._weight_noise_allowed)]
        min_wffn1_cost = self._wffn1_noise_cost_map[min(self._wffn1_noise_allowed)]
        max_wffn1_cost = self._wffn1_noise_cost_map[max(self._wffn1_noise_allowed)]
        mean_input_cost = np.mean([self._input_noise_cost_map[sf] for sf in self._input_noise_allowed])
        mean_generic_weight_cost = np.mean(
            [self._weight_noise_cost_map[sf] for sf in self._weight_noise_allowed]
        )
        mean_wffn1_cost = np.mean(
            [self._wffn1_noise_cost_map[sf] for sf in self._wffn1_noise_allowed]
        )
        self.min_cost_per_layer = min_input_cost + 5 * min_generic_weight_cost + min_wffn1_cost
        self.max_cost_per_layer = max_input_cost + 5 * max_generic_weight_cost + max_wffn1_cost
        self.expected_cost_per_layer = mean_input_cost + 5 * mean_generic_weight_cost + mean_wffn1_cost
        self._cost_lower_bound = self.min_cost_per_layer * total_layers
        self._cost_upper_bound = float(self.baseline_cost)

        # \u65b9\u5dee\u503a\u52a1\u5f52\u4e00\u5316\uff1alog2 \u6620\u5c04\u533a\u95f4 [\u5168\u5c40\u6700\u5c0f, \u5168\u5c40\u6700\u5927]
        max_single_layer_debt = _compute_max_single_layer_debt()
        min_single_layer_debt = _compute_min_single_layer_debt()
        self._max_total_variance_debt = max_single_layer_debt * total_layers
        self._min_total_variance_debt = min_single_layer_debt * total_layers

        self.current_episode_metrics = None
        self.last_reward_components = None
        self.last_mc_eval = None
        self.reset()

    def reset(self):
        self.current_layer = 0
        self.accumulated_cost = 0.0
        self.accumulated_noise_variance = 0.0
        self.current_episode_metrics = None
        self.last_reward_components = None
        self.last_mc_eval = None

        self.input_noise_config = []
        self.wq_noise_config = []
        self.wk_noise_config = []
        self.wv_noise_config = []
        self.wo_noise_config = []
        self.wffn1_noise_config = []
        self.wffn2_noise_config = []

        self.prev_action_indices = np.array(self._sos_tokens, dtype=np.int64)
        self.prev_scalings = {
            "x": max(self._input_noise_allowed),
            "wq": max(self._weight_noise_allowed),
            "wk": max(self._weight_noise_allowed),
            "wv": max(self._weight_noise_allowed),
            "wo": max(self._weight_noise_allowed),
            "wffn1": max(self._wffn1_noise_allowed),
            "wffn2": max(self._weight_noise_allowed),
        }

        self.input_history = np.full(self.total_layers, self._history_mask, dtype=np.float32)
        self.wq_history = np.full(self.total_layers, self._history_mask, dtype=np.float32)
        self.wk_history = np.full(self.total_layers, self._history_mask, dtype=np.float32)
        self.wv_history = np.full(self.total_layers, self._history_mask, dtype=np.float32)
        self.wo_history = np.full(self.total_layers, self._history_mask, dtype=np.float32)
        self.wffn1_history = np.full(self.total_layers, self._history_mask, dtype=np.float32)
        self.wffn2_history = np.full(self.total_layers, self._history_mask, dtype=np.float32)
        return self._get_state()

    def set_episode_progress(self, episode, total_episodes):
        self._episode_progress = float(episode) / max(1, int(total_episodes))

    def get_continuous_features(self):
        """V2 5 \u7ef4\u8fde\u7eed\u7279\u5f81\u3002"""
        # 1. cost_deviation [-1, 1]
        expected_cost_so_far = self.current_layer * self.expected_cost_per_layer
        if expected_cost_so_far > 0:
            cost_deviation = (self.accumulated_cost - expected_cost_so_far) / expected_cost_so_far
        else:
            cost_deviation = 0.0
        cost_deviation = np.clip(cost_deviation, -1.0, 1.0)

        # 2. complexity_debt [0, 1]
        baseline_cost_so_far = self.current_layer * self.max_cost_per_layer
        if baseline_cost_so_far > 0:
            complexity_debt = (baseline_cost_so_far - self.accumulated_cost) / baseline_cost_so_far
        else:
            complexity_debt = 0.0
        complexity_debt = np.clip(complexity_debt, 0.0, 1.0)

        # 3. progress [0, 1]
        progress = self.current_layer / self.total_layers

        # 4. noise_debt_ratio [0, 1]：log2 将极小 CKKS 方差映射到 [0,1]，避免 log(1+alpha*x)≈x 导致的特征塌陷
        if self._max_total_variance_debt > 0:
            acc = float(self.accumulated_noise_variance)
            if acc <= 0.0:
                noise_debt_ratio = 0.0
            else:
                lo = math.log2(max(self._min_total_variance_debt, 1e-300))
                hi = math.log2(max(self._max_total_variance_debt, 1e-300))
                v = math.log2(max(acc, 1e-300))
                denom = hi - lo
                if denom <= 1e-12:
                    noise_debt_ratio = 1.0 if v >= hi else 0.0
                else:
                    noise_debt_ratio = (v - lo) / denom
        else:
            noise_debt_ratio = 0.0
        noise_debt_ratio = float(np.clip(noise_debt_ratio, 0.0, 1.0))

        # 5. remaining_budget_ratio [0, 1]
        remaining_budget_ratio = float(np.clip(
            (float(self._cost_upper_bound) - float(self.accumulated_cost))
            / max(float(self._cost_upper_bound) - float(self._cost_lower_bound), 1e-8),
            0.0, 1.0,
        ))

        return np.array(
            [cost_deviation, complexity_debt, progress, noise_debt_ratio, remaining_budget_ratio],
            dtype=np.float32,
        )

    def _get_state(self):
        position = np.zeros(self.total_layers, dtype=np.float32)
        if self.current_layer < self.total_layers:
            position[self.current_layer] = 1.0

        prev_norms = np.array([
            self._input_noise_to_norm[self.prev_scalings["x"]],
            self._weight_noise_to_norm[self.prev_scalings["wq"]],
            self._weight_noise_to_norm[self.prev_scalings["wk"]],
            self._weight_noise_to_norm[self.prev_scalings["wv"]],
            self._weight_noise_to_norm[self.prev_scalings["wo"]],
            self._wffn1_noise_to_norm[self.prev_scalings["wffn1"]],
            self._weight_noise_to_norm[self.prev_scalings["wffn2"]],
        ], dtype=np.float32)

        cont = self.get_continuous_features()
        state = np.concatenate([
            position,
            [cont[0]],
            prev_norms,
            [cont[1]],
            [cont[2]],
            self.input_history,
            self.wq_history,
            self.wk_history,
            self.wv_history,
            self.wo_history,
            self.wffn1_history,
            self.wffn2_history,
            cont[3:],
        ])
        return state.astype(np.float32)

    def step(self, input_action_idx, wq_action_idx, wk_action_idx, wv_action_idx,
             wo_action_idx, wffn1_action_idx, wffn2_action_idx):
        input_sf = self._input_noise_scaling_map[int(input_action_idx)]
        wq_sf = self._wq_map[int(wq_action_idx)]
        wk_sf = self._wk_map[int(wk_action_idx)]
        wv_sf = self._wv_map[int(wv_action_idx)]
        wo_sf = self._wo_map[int(wo_action_idx)]
        wffn1_sf = self._wffn1_map[int(wffn1_action_idx)]
        wffn2_sf = self._wffn2_map[int(wffn2_action_idx)]

        self.input_noise_config.append(input_sf)
        self.wq_noise_config.append(wq_sf)
        self.wk_noise_config.append(wk_sf)
        self.wv_noise_config.append(wv_sf)
        self.wo_noise_config.append(wo_sf)
        self.wffn1_noise_config.append(wffn1_sf)
        self.wffn2_noise_config.append(wffn2_sf)

        step_cost = (
            self._input_noise_cost_map[input_sf]
            + self._weight_noise_cost_map[wq_sf]
            + self._weight_noise_cost_map[wk_sf]
            + self._weight_noise_cost_map[wv_sf]
            + self._weight_noise_cost_map[wo_sf]
            + self._wffn1_noise_cost_map[wffn1_sf]
            + self._weight_noise_cost_map[wffn2_sf]
        )
        self.accumulated_cost += step_cost

        # \u65b9\u5dee\u503a\u52a1\u7d2f\u52a0 (CKKS: variance ~ 2^(-2*exp))
        step_variance_debt = (
            _compute_ckks_variance(input_sf)
            + _compute_ckks_variance(wq_sf)
            + _compute_ckks_variance(wk_sf)
            + _compute_ckks_variance(wv_sf)
            + _compute_ckks_variance(wo_sf)
            + _compute_ckks_variance(wffn1_sf)
            + _compute_ckks_variance(wffn2_sf)
        )
        self.accumulated_noise_variance += step_variance_debt

        self.prev_action_indices = np.array([
            int(input_action_idx), int(wq_action_idx), int(wk_action_idx),
            int(wv_action_idx), int(wo_action_idx), int(wffn1_action_idx),
            int(wffn2_action_idx),
        ], dtype=np.int64)
        self.prev_scalings = {
            "x": input_sf, "wq": wq_sf, "wk": wk_sf, "wv": wv_sf,
            "wo": wo_sf, "wffn1": wffn1_sf, "wffn2": wffn2_sf,
        }

        self.input_history[self.current_layer] = self._input_noise_to_norm[input_sf]
        self.wq_history[self.current_layer] = self._weight_noise_to_norm[wq_sf]
        self.wk_history[self.current_layer] = self._weight_noise_to_norm[wk_sf]
        self.wv_history[self.current_layer] = self._weight_noise_to_norm[wv_sf]
        self.wo_history[self.current_layer] = self._weight_noise_to_norm[wo_sf]
        self.wffn1_history[self.current_layer] = self._wffn1_noise_to_norm[wffn1_sf]
        self.wffn2_history[self.current_layer] = self._weight_noise_to_norm[wffn2_sf]

        info = {
            "layer_index": self.current_layer,
            "curr_input_noise_scaling_factor": input_sf,
            "curr_wq_noise_scaling_factor": wq_sf,
            "curr_wk_noise_scaling_factor": wk_sf,
            "curr_wv_noise_scaling_factor": wv_sf,
            "curr_wo_noise_scaling_factor": wo_sf,
            "curr_wffn1_noise_scaling_factor": wffn1_sf,
            "curr_wffn2_noise_scaling_factor": wffn2_sf,
            "accumulated_cost": self.accumulated_cost,
            "accumulated_noise_variance": self.accumulated_noise_variance,
            "step_variance_debt": step_variance_debt,
            "input_noise_config": self.input_noise_config.copy(),
            "wq_noise_config": self.wq_noise_config.copy(),
            "wk_noise_config": self.wk_noise_config.copy(),
            "wv_noise_config": self.wv_noise_config.copy(),
            "wo_noise_config": self.wo_noise_config.copy(),
            "wffn1_noise_config": self.wffn1_noise_config.copy(),
            "wffn2_noise_config": self.wffn2_noise_config.copy(),
        }

        self.current_layer += 1
        if self.current_layer < self.total_layers:
            # \u975e\u7ec8\u6b62\u6b65\uff1areward = 0, done = False
            info["step_reward"] = 0.0
            info["raw_final_reward"] = None
            info["final_selection_score"] = None
            info["mc_eval"] = None
            return self._get_state(), 0.0, False, info

        # \u7ec8\u6b62\u6b65\uff1aMC \u8bc4\u4f30 + \u7ec8\u7aef\u5956\u52b1
        terminal_reward, reward_components, mc_eval = self._compute_terminal_reward_mc()
        self.current_episode_metrics = {
            "loss": mc_eval.get("loss_mean", 0.0),
            "metric1": mc_eval.get("p_mean", 0.0),
            "metric2": mc_eval.get("s_mean", 0.0),
            "cost": self.accumulated_cost,
        }
        self.last_reward_components = dict(reward_components)
        self.last_mc_eval = mc_eval

        info["step_reward"] = terminal_reward
        info["raw_final_reward"] = terminal_reward
        info["final_selection_score"] = reward_components.get("final_selection_score", terminal_reward)
        info["mc_eval"] = mc_eval
        info["reward_components"] = reward_components
        return self._get_state(), terminal_reward, True, info

    def _compute_terminal_reward_mc(self):
        """MC 终端奖励：将验证集切成 N 段、每段独立噪声采样评测，总计只遍历 1 遍数据。"""
        noise_kwargs = {
            "input_noise_scaling_factors": np.array(self.input_noise_config, dtype=int),
            "wq_noise_scaling_factors": np.array(self.wq_noise_config, dtype=int),
            "wk_noise_scaling_factors": np.array(self.wk_noise_config, dtype=int),
            "wv_noise_scaling_factors": np.array(self.wv_noise_config, dtype=int),
            "wo_noise_scaling_factors": np.array(self.wo_noise_config, dtype=int),
            "wffn1_noise_scaling_factors": np.array(self.wffn1_noise_config, dtype=int),
            "wffn2_noise_scaling_factors": np.array(self.wffn2_noise_config, dtype=int),
        }

        mc_eval = self.evaluator.evaluate_noise_model_segmented(
            segments=self._mc_train_samples,
            **noise_kwargs,
        )

        loss_mean = float(mc_eval["loss_mean"])
        loss_std = float(mc_eval["loss_std"])
        m1_mean = float(mc_eval["p_mean"])
        m1_std = float(mc_eval["p_std"])
        m2_mean = float(mc_eval["s_mean"])
        m2_std = float(mc_eval["s_std"])

        # -- \u6027\u80fd\u5206\u6570 (margin-based) --
        limits = self.constraint_limits
        _MARGIN_DENOM_FLOOR = 0.01
        margin_loss = (float(limits["loss"]) - loss_mean) / max(
            abs(float(limits["loss"]) - float(self.baseline_loss)), _MARGIN_DENOM_FLOOR
        )
        margin_m1 = (m1_mean - float(limits["metric1"])) / max(
            abs(float(self.baseline_p) - float(limits["metric1"])), _MARGIN_DENOM_FLOOR
        )
        if self.num_metrics > 1:
            margin_m2 = (m2_mean - float(limits["metric2"])) / max(
                abs(float(self.baseline_s) - float(limits["metric2"])), _MARGIN_DENOM_FLOOR
            )
        else:
            margin_m2 = 0.0

        # violation penalty
        viol_loss = max(0.0, -margin_loss)
        viol_m1 = max(0.0, -margin_m1)
        viol_m2 = max(0.0, -margin_m2) if self.num_metrics > 1 else 0.0
        violation_penalty = (
            float(self._barrier_weights["loss"]) * viol_loss
            + float(self._barrier_weights["metric1"]) * viol_m1
        )
        if self.num_metrics > 1:
            violation_penalty += float(self._barrier_weights.get("metric2", 0.0)) * viol_m2

        # margin utility (perf_score)
        util_loss = max(0.0, margin_loss)
        util_m1 = max(0.0, margin_m1)
        util_m2 = max(0.0, margin_m2) if self.num_metrics > 1 else 0.0
        perf_score = (
            float(self._perf_weights["loss"]) * util_loss
            + float(self._perf_weights["metric1"]) * util_m1
        )
        if self.num_metrics > 1:
            perf_score += float(self._perf_weights.get("metric2", 0.0)) * util_m2

        # -- \u6210\u672c\u5206\u6570 --
        cost_score = float(np.clip(
            (float(self._cost_upper_bound) - float(self.accumulated_cost))
            / max(float(self._cost_upper_bound) - float(self._cost_lower_bound), 1e-8),
            0.0, 1.0,
        ))

        # -- \u65b9\u5dee\u60e9\u7f5a (Stability Penalty) --
        # sigma_cand: \u5019\u9009\u65b9\u6848\u7684 MC \u65b9\u5dee\u7684\u52a0\u6743\u5747\u503c
        sigma_components = [loss_std, m1_std]
        if self.num_metrics > 1:
            sigma_components.append(m2_std)
        sigma_cand = float(np.mean(sigma_components))

        sigma_base_components = [self._baseline_loss_std, self._baseline_m1_std]
        sigma_base = float(np.mean(sigma_base_components))

        # 超出「baseline–worst 动态 std 上界」的部分才惩罚（与 confirm 口径一致）
        _excess = [
            max(0.0, float(loss_std) - self._dynamic_loss_std_cap),
            max(0.0, float(m1_std) - self._dynamic_m1_std_cap),
        ]
        if self.num_metrics > 1:
            _excess.append(max(0.0, float(m2_std) - self._dynamic_m2_std_cap))
        stability_penalty = -self._stability_lambda * float(np.mean(_excess))

        # -- \u7ec8\u7aef\u5956\u52b1 --
        terminal_reward = (
            float(NOISE_STAGE_TAIL_MARGIN_WEIGHT) * perf_score
            - float(NOISE_STAGE_TAIL_VIOLATION_WEIGHT) * violation_penalty
            + float(NOISE_STAGE_COST_WEIGHT) * cost_score
            + stability_penalty
        )

        # \u5b89\u5168\u88c1\u526a
        clipped_reward = float(np.clip(
            terminal_reward, self._reward_clip_min, self._reward_clip_max,
        ))

        constraints_ok = (margin_loss >= 0.0) and (margin_m1 >= 0.0)
        if self.num_metrics > 1:
            constraints_ok = constraints_ok and (margin_m2 >= 0.0)

        reward_components = {
            "loss_limit": float(limits["loss"]),
            "metric1_limit": float(limits["metric1"]),
            "metric2_limit": float(limits.get("metric2", limits["metric1"])),
            "margin_loss": float(margin_loss),
            "margin_m1": float(margin_m1),
            "margin_m2": float(margin_m2),
            "perf_score": float(perf_score),
            "cost_score": float(cost_score),
            "violation_penalty": float(violation_penalty),
            "stability_penalty": float(stability_penalty),
            "sigma_cand": float(sigma_cand),
            "sigma_base": float(sigma_base),
            "constraints_ok": bool(constraints_ok),
            "cost_lower_bound": float(self._cost_lower_bound),
            "cost_upper_bound": float(self._cost_upper_bound),
            "current_cost": float(self.accumulated_cost),
            "accumulated_noise_variance": float(self.accumulated_noise_variance),
            "mc_train_samples": int(self._mc_train_samples),
            "loss_mean": float(loss_mean),
            "loss_std": float(loss_std),
            "m1_mean": float(m1_mean),
            "m1_std": float(m1_std),
            "m2_mean": float(m2_mean),
            "m2_std": float(m2_std),
            "raw_terminal_reward": float(terminal_reward),
            "raw_final_reward": float(clipped_reward),
            "final_selection_score": float(clipped_reward),
            "terminal_reward_mode": "mc_sparse_v2",
        }
        return clipped_reward, reward_components, mc_eval


# ===========================================================================
# PPO Update (V2: \u5355 critic, \u65e0 mean_aux_head)
# ===========================================================================

def _ppo_update_noise_gtrxl(evaluator, noise_net, optimizer, buffer, device,
                            entropy_coef=None,
                            ppo_update_step=0,
                            ppo_eps_clip=0.12, ppo_k_epochs=6,
                            ppo_value_coef=0.5,
                            gtrxl_warmup_mode="constant",
                            gtrxl_warmup_updates=0,
                            gtrxl_short_warmup_updates=20,
                            gtrxl_entropy_lower_bound=0.005,
                            gtrxl_mini_batch_episodes=8,
                            value_clip_range=0.2):
    """V2 PPO \u66f4\u65b0\uff1a\u5355 critic\uff0c\u65e0 mean_aux_head\u3002"""
    if entropy_coef is None:
        entropy_coef = evaluator.get_current_entropy_coef()

    target_lr = float(evaluator.ppo_lr_initial)
    warmup_updates = max(0, int(gtrxl_warmup_updates))
    if gtrxl_warmup_mode == "short":
        warmup_updates = max(1, int(gtrxl_short_warmup_updates))

    if gtrxl_warmup_mode == "constant" or warmup_updates <= 0:
        current_lr = target_lr
    elif ppo_update_step < warmup_updates:
        warmup_factor = float(ppo_update_step + 1) / float(warmup_updates)
        current_lr = target_lr * warmup_factor
    else:
        current_lr = target_lr

    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    (cont_features, layer_indices, prev_actions, actions,
     old_logprobs, rewards, values, dones) = buffer.get_batch(device)

    n_eps = cont_features.size(0)
    all_advantages = []
    all_returns = []
    for i in range(n_eps):
        adv, ret = evaluator.compute_gae(
            rewards[i].cpu().numpy(),
            values[i].cpu().numpy(),
            dones[i].cpu().numpy(),
            gamma=NOISE_STAGE_PPO_GAMMA,
        )
        all_advantages.append(adv)
        all_returns.append(ret)

    advantages = torch.stack(all_advantages).to(device)
    returns = torch.stack(all_returns).to(device)
    adv_flat = advantages.reshape(-1)
    advantages = (advantages - adv_flat.mean()) / (adv_flat.std() + 1e-8)

    evaluator.return_normalizer.update(returns)
    returns_normalized = torch.tensor(
        evaluator.return_normalizer.normalize(returns.cpu().numpy()),
        dtype=torch.float32,
    ).to(device)
    values_normalized = torch.tensor(
        evaluator.return_normalizer.normalize(values.cpu().numpy()),
        dtype=torch.float32,
    ).to(device)

    last_policy_loss = 0.0
    last_value_loss = 0.0
    last_entropy = 0.0

    for _ in range(ppo_k_epochs):
        ep_indices = torch.randperm(n_eps)
        for start in range(0, n_eps, gtrxl_mini_batch_episodes):
            end = min(start + gtrxl_mini_batch_episodes, n_eps)
            mb_idx = ep_indices[start:end]

            mb_cont = cont_features[mb_idx]
            mb_layer = layer_indices[mb_idx]
            mb_prev_actions = prev_actions[mb_idx]
            mb_actions = actions[mb_idx]
            mb_old_lp = old_logprobs[mb_idx]
            mb_adv = advantages[mb_idx]
            mb_ret = returns_normalized[mb_idx]
            mb_old_val = values_normalized[mb_idx]

            new_logprobs, entropy, new_values = noise_net.evaluate_actions(
                mb_cont, mb_layer, mb_prev_actions, mb_actions
            )

            new_logprobs_flat = new_logprobs.reshape(-1)
            entropy_flat = entropy.reshape(-1)
            new_val_flat = new_values.reshape(-1)
            mb_old_lp_flat = mb_old_lp.reshape(-1)
            mb_adv_flat = mb_adv.reshape(-1)
            mb_ret_flat = mb_ret.reshape(-1)
            mb_old_val_flat = mb_old_val.reshape(-1)

            # \u7b56\u7565\u635f\u5931
            ratios = torch.exp(new_logprobs_flat - mb_old_lp_flat)
            surr1 = ratios * mb_adv_flat
            surr2 = torch.clamp(ratios, 1 - ppo_eps_clip, 1 + ppo_eps_clip) * mb_adv_flat
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value head \u635f\u5931 (clipped + Huber)
            new_val_norm = (
                (new_val_flat - evaluator.return_normalizer.mean)
                / (evaluator.return_normalizer.std + 1e-8)
            )
            value_clipped = mb_old_val_flat + torch.clamp(
                new_val_norm - mb_old_val_flat,
                -value_clip_range, value_clip_range,
            )
            huber_loss_fn = nn.HuberLoss(reduction="none", delta=1.0)
            vl_unclipped = huber_loss_fn(new_val_norm, mb_ret_flat)
            vl_clipped = huber_loss_fn(value_clipped, mb_ret_flat)
            value_loss = torch.max(vl_unclipped, vl_clipped).mean()

            mean_entropy = entropy_flat.mean()
            effective_entropy_coef = entropy_coef
            if mean_entropy.item() < gtrxl_entropy_lower_bound:
                entropy_deficit = gtrxl_entropy_lower_bound - mean_entropy.item()
                effective_entropy_coef = entropy_coef + 10.0 * entropy_deficit

            entropy_loss = -mean_entropy
            loss = (
                policy_loss
                + ppo_value_coef * value_loss
                + effective_entropy_coef * entropy_loss
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(noise_net.parameters(), 0.5)
            optimizer.step()

            last_policy_loss = policy_loss.item()
            last_value_loss = value_loss.item()
            last_entropy = mean_entropy.item()

    return last_policy_loss, last_value_loss, last_entropy


# ===========================================================================
# Plot functions
# ===========================================================================

def _plot_noise_training_curves(
    evaluator,
    episode_returns, episode_raw_final_rewards,
    episode_losses, episode_metric1s, episode_metric2s,
    episode_entropies,
    base_loss, base_p, base_s,
    training_curve_path="noise_ppo_training_curve.png",
    entropy_curve_path="noise_ppo_entropy_curve.png",
    ppo_update_interval=170,
):
    if len(episode_returns) == 0:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        episodes = np.arange(1, len(episode_returns) + 1)
        episode_returns_arr = np.array(episode_returns, dtype=np.float32)
        raw_final_rewards = np.array(episode_raw_final_rewards, dtype=np.float32)
        losses = np.array(episode_losses, dtype=np.float32)
        metric1s = np.array(episode_metric1s, dtype=np.float32)
        metric2s = np.array(episode_metric2s, dtype=np.float32)
        metric_names_tuple = evaluator.get_metric_names()
        _num_m = evaluator.get_num_metrics()
        _m1_name = metric_names_tuple[0]
        _m2_name = metric_names_tuple[1] if _num_m > 1 else metric_names_tuple[0]
        window = min(50, max(1, len(episode_returns_arr) // 10))

        def compute_ma(data):
            if len(data) < window:
                return data
            kernel = np.ones(window, dtype=np.float32) / window
            return np.convolve(data, kernel, mode="valid")

        episode_returns_ma = compute_ma(episode_returns_arr)
        raw_final_rewards_ma = compute_ma(raw_final_rewards)
        losses_ma = compute_ma(losses)
        metric1s_ma = compute_ma(metric1s)
        metric2s_ma = compute_ma(metric2s) if _num_m > 1 else None
        episodes_ma = episodes[window - 1:] if len(episode_returns_arr) >= window else episodes

        if _num_m == 1:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle("Noise PPO V2 Training Curves", fontsize=14, fontweight="bold")
            ax1, ax2, ax3 = axes
            ax4 = None
        else:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle("Noise PPO V2 Training Curves", fontsize=14, fontweight="bold")
            ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

        ax1.plot(episodes, episode_returns_arr, label="Episode Return", alpha=0.45, color="steelblue")
        ax1.plot(episodes_ma, episode_returns_ma, label=f"MA ({window})", linewidth=2, color="navy")
        ax1.plot(episodes, raw_final_rewards, label="Raw Final Reward", alpha=0.45, color="darkorange")
        ax1.plot(episodes_ma, raw_final_rewards_ma, label=f"Final MA ({window})", linewidth=2, color="orangered")
        ax1.set_xlabel("Episode"); ax1.set_ylabel("Reward"); ax1.set_title("Return vs Final Reward"); ax1.grid(True, alpha=0.3); ax1.legend()

        ax2.plot(episodes, losses, label="Loss", alpha=0.6, color="red")
        ax2.plot(episodes_ma, losses_ma, label=f"MA ({window})", linewidth=2, color="darkred")
        ax2.axhline(y=base_loss, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Baseline")
        ax2.set_xlabel("Episode"); ax2.set_ylabel("Loss"); ax2.set_title("Loss"); ax2.grid(True, alpha=0.3); ax2.legend()

        ax3.plot(episodes, metric1s, label=_m1_name, alpha=0.6, color="green")
        ax3.plot(episodes_ma, metric1s_ma, label=f"MA ({window})", linewidth=2, color="darkgreen")
        ax3.axhline(y=base_p, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Baseline")
        ax3.set_xlabel("Episode"); ax3.set_ylabel(_m1_name); ax3.set_title(_m1_name); ax3.grid(True, alpha=0.3); ax3.legend()

        if ax4 is not None:
            ax4.plot(episodes, metric2s, label=_m2_name, alpha=0.6, color="purple")
            ax4.plot(episodes_ma, metric2s_ma, label=f"MA ({window})", linewidth=2, color="darkviolet")
            ax4.axhline(y=base_s, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Baseline")
            ax4.set_xlabel("Episode"); ax4.set_ylabel(_m2_name); ax4.set_title(_m2_name); ax4.grid(True, alpha=0.3); ax4.legend()

        plt.tight_layout()
        training_dir = os.path.dirname(training_curve_path)
        if training_dir:
            os.makedirs(training_dir, exist_ok=True)
        plt.savefig(training_curve_path, dpi=150)
        plt.close()
        evaluator.log(f"  [\u7ed8\u56fe] \u566a\u58f0 PPO V2 \u8bad\u7ec3\u66f2\u7ebf\u5df2\u5199\u5165 path={training_curve_path}")

        if episode_entropies:
            update_episodes = np.arange(ppo_update_interval, len(episode_returns) + 1, ppo_update_interval)
            entropies = np.array(episode_entropies, dtype=np.float32)
            if len(update_episodes) == len(entropies):
                fig_ent, ax_ent = plt.subplots(1, 1, figsize=(10, 5))
                ax_ent.plot(update_episodes, entropies, label="Policy Entropy", alpha=0.8, color="teal", marker="o", markersize=3)
                ax_ent.set_xlabel("Episode"); ax_ent.set_ylabel("Entropy"); ax_ent.set_title("Policy Entropy")
                ax_ent.grid(True, alpha=0.3); ax_ent.legend()
                plt.tight_layout()
                entropy_dir = os.path.dirname(entropy_curve_path)
                if entropy_dir:
                    os.makedirs(entropy_dir, exist_ok=True)
                plt.savefig(entropy_curve_path, dpi=150)
                plt.close()
                evaluator.log(f"  [\u7ed8\u56fe] \u566a\u58f0 PPO V2 \u71b5\u66f2\u7ebf\u5df2\u5199\u5165 path={entropy_curve_path}")
    except Exception as e:
        evaluator.log(f"  [\u8b66\u544a] \u7ed8\u5236\u566a\u58f0 PPO V2 \u8bad\u7ec3\u66f2\u7ebf\u5931\u8d25: {e}")


# ===========================================================================
# NoiseRLModuleV2
# ===========================================================================

class NoiseRLModuleV2:
    """V2 \u4e8c\u9636\u6bb5\u566a\u58f0 RL \u6a21\u5757\u3002

    \u7b80\u5316\uff1a\u7a00\u758f\u5956\u52b1 + \u566a\u58f0\u65b9\u5dee\u503a\u52a1 + MC \u7ec8\u7aef\u5956\u52b1 + \u5355 critic + \u786e\u8ba4\u5931\u8d25\u60e9\u7f5a\u3002
    """

    def __init__(self, evaluator):
        self.evaluator = evaluator

    def run(self, fixed_gelu, fixed_softmax, fixed_label, fixed_source,
             resume_checkpoint_path=None):
        from layer_importance_evaluator import (
            INPUT_NOISE_SCALING_MAP,
            INPUT_NOISE_COST_MAP,
            INPUT_NOISE_SCALING_TO_NORM,
            WEIGHT_NOISE_COST_MAP,
            WEIGHT_NOISE_SCALING_TO_NORM,
            WFFN1_NOISE_COST_MAP,
            WFFN1_NOISE_SCALING_TO_NORM,
            WQ_NOISE_SCALING_MAP,
            WK_NOISE_SCALING_MAP,
            WV_NOISE_SCALING_MAP,
            WO_NOISE_SCALING_MAP,
            WFFN1_NOISE_SCALING_MAP,
            WFFN2_NOISE_SCALING_MAP,
            NOISE_STAGE_NUM_ACTIONS,
            NOISE_STAGE_SOS_TOKENS,
            NOISE_STAGE_PREV_ACTION_EMBED_DIM,
            NOISE_STAGE_ACTION_DIMS,
            NOISE_STAGE_TRAINING_CURVE_PATH,
            NOISE_STAGE_ENTROPY_CURVE_PATH,
            PPO_UPDATE_INTERVAL,
            PPO_VALUE_COEF,
            REWARD_CLIP_MIN,
            REWARD_CLIP_MAX,
            HISTORY_MASK_VALUE,
            LSTM_POS_DIM,
            LSTM_PROJ_DIM,
            GTRXL_ENTROPY_LOWER_BOUND,
            GTRXL_MINI_BATCH_EPISODES,
            GTrXLBlock,
            STEP_INFO_CHUNK_SIZE,
            REWARD_DROP_WARNING_THRESHOLD,
        )

        ev = self.evaluator
        noise_progress_dir = getattr(ev, "noise_stage_progress_dir", NOISE_STAGE_PROGRESS_DIR)
        noise_training_curve_path = getattr(ev, "noise_stage_training_curve_path", NOISE_STAGE_TRAINING_CURVE_PATH)
        noise_entropy_curve_path = getattr(ev, "noise_stage_entropy_curve_path", NOISE_STAGE_ENTROPY_CURVE_PATH)
        stage2_total_episodes = int(getattr(ev, "stage2_rl_episodes", NOISE_STAGE_PPO_MAX_EPISODES))
        _num_metrics_for_log = ev.get_num_metrics()
        _metric_names_for_log = ev.get_metric_names()

        _noise_log_major_rule(
            ev.log,
            "\u9636\u6bb55 \u00b7 \u4e8c\u9636\u6bb5\u566a\u58f0\u5f3a\u5316\u5b66\u4e60 V2\uff08Phase 5 \u00b7 Second-stage noise RL V2\uff09",
        )
        ev.log(
            f"  \u25b8 \u56fa\u5b9a GELU / Softmax \u6765\u6e90\uff08source\uff09: {fixed_source}    "
            f"\u6807\u7b7e\uff08label\uff09: {fixed_label}"
        )
        ev.log(f"  \u25b8 GELU \u79bb\u6563\u9636\u6570\u5411\u91cf:   {np.asarray(fixed_gelu, dtype=int).tolist()}")
        ev.log(f"  \u25b8 Softmax \u79bb\u6563\u9636\u6570\u5411\u91cf: {np.asarray(fixed_softmax, dtype=int).tolist()}")
        ev.log(
            f"  \u25b8 \u6307\u6807\u7ef4\u5ea6 num_metrics={_num_metrics_for_log}    "
            f"\u6307\u6807\u540d metric names={_metric_names_for_log}"
        )

        fixed_gelu = np.asarray(fixed_gelu, dtype=int)
        fixed_softmax = np.asarray(fixed_softmax, dtype=int)
        cost_reference_noise_config = ev._get_max_noise_configuration()

        baseline_noise_config = _get_low_risk_noise_configuration(ev)
        worst_case_noise_config = _get_worst_case_noise_configuration(ev)

        # \u4e8c\u9636\u6bb5\u566a\u58f0 RL \u7edf\u4e00\u5728\u5b8c\u6574\u9a8c\u8bc1\u96c6\u4e0a\u8bc4\u6d4b\uff0c\u4e0d\u4f7f\u7528 val_search_full / val_holdout \u5212\u5206\u3002
        if ev.has_dataset_split("validation_full"):
            reward_reference_split = "validation_full"
        else:
            reward_reference_split = ev.get_reward_reference_split_name()
        ev.log(
            f"  \u25b8 Noise RL \u8bc4\u6d4b\u5b50\u96c6 split=\u300c{reward_reference_split}\u300d"
            + (
                "\uff08\u5b8c\u6574\u9a8c\u8bc1\u96c6 validation_full\uff09"
                if reward_reference_split == "validation_full"
                else "\uff08\u56de\u9000\uff1a\u672a\u6ce8\u518c validation_full\uff0c\u4f7f\u7528 get_reward_reference_split_name\uff09"
            )
        )

        def _copy_repeat_summary(summary):
            return {
                key: ([dict(item) for item in value] if key == "trials" else value)
                for key, value in summary.items()
            }

        def _log_repeat_baseline(label, stats):
            ev.log(
                f"{label} \u00b7 \u5206\u6bb5 segments={stats['n']} \u00b7 \u5b50\u96c6 split={stats['split_name']}"
            )
            ev.log(
                "     "
                f"{ev._fmt_metrics(stats['loss_mean'], stats['p_mean'], stats['s_mean'])}  "
                f"\u2502 \u6807\u51c6\u5dee std: loss={stats['loss_std']:.4f}, m1={stats['p_std']:.4f}, m2={stats['s_std']:.4f}"
            )

        _eval_dataset = ev.dataset_splits.get(reward_reference_split)
        _eval_dataset_size = len(_eval_dataset) if _eval_dataset is not None else 0
        baseline_segments = _auto_adjust_segments(
            _eval_dataset_size, NOISE_STAGE_BASELINE_SEGMENTS, log_fn=ev.log,
        )

        baseline_reference_stats = ev.evaluate_model_with_attention_noise_segmented(
            fixed_gelu, fixed_softmax,
            **baseline_noise_config,
            segments=baseline_segments,
            split=reward_reference_split,
        )
        # Worst-case \u8bc4\u4f30
        worst_reference_stats = ev.evaluate_model_with_attention_noise_segmented(
            fixed_gelu, fixed_softmax,
            **worst_case_noise_config,
            segments=baseline_segments,
            split=reward_reference_split,
        )
        cost_reference_tot_c, cost_reference_breakdown = ev.get_noise_simulated_cost(**cost_reference_noise_config)

        _noise_block_title(ev.log, "\u6027\u80fd\u57fa\u7ebf \u00b7 \u56fa\u5b9a Stage-1 + \u4f4e\u98ce\u9669\u53c2\u8003\u566a\u58f0\uff08Noise-stage baseline\uff09")
        ev.log("  \u25b8 \u4f4e\u98ce\u9669\u7f29\u653e\u56e0\u5b50\u914d\u7f6e\uff08low-risk noise config\uff09\uff1a")
        for _bk, _bv in baseline_noise_config.items():
            ev.log(f"     {_bk}: {np.asarray(_bv, dtype=int).tolist()}")
        _log_repeat_baseline("  \u57fa\u7ebf\uff08Baseline\uff09", baseline_reference_stats)

        _noise_block_title(ev.log, "\u6700\u574f\u60c5\u5f62 \u00b7 \u56fa\u5b9a Stage-1 + \u6700\u5f3a\u566a\u58f0\uff08Worst-case noise\uff09")
        _log_repeat_baseline("  \u6700\u574f\uff08Worst\uff09", worst_reference_stats)

        _noise_block_title(ev.log, "\u6210\u672c\u53c2\u8003\uff08Cost reference\uff09")
        ev.log(f"  \u25b8 \u805a\u5408\u6210\u672c\uff08noise cost\uff09: {cost_reference_tot_c:.2f}")

        base_loss = baseline_reference_stats["loss_mean"]
        base_p = baseline_reference_stats["p_mean"]
        base_s = baseline_reference_stats["s_mean"]
        worst_loss = worst_reference_stats["loss_mean"]
        worst_p = worst_reference_stats["p_mean"]
        worst_s = worst_reference_stats["s_mean"]

        # \u52a8\u6001\u7ea6\u675f
        search_limits = _compute_dynamic_limits(base_loss, base_p, base_s, worst_loss, worst_p, worst_s)
        limit_loss = float(search_limits["loss"])
        limit_p = float(search_limits["metric1"])
        limit_s = float(search_limits["metric2"])

        _noise_block_title(
            ev.log,
            f"\u52a8\u6001\u7ea6\u675f\uff08Dynamic constraints, quartile={NOISE_STAGE_DYNAMIC_LIMIT_QUARTILE} on {reward_reference_split}\uff09",
        )
        ev.log(f"  \u25b8 \u57fa\u7ebf\uff08baseline\uff09: {ev._fmt_metrics(base_loss, base_p, base_s)}")
        ev.log(f"  \u25b8 \u6700\u574f\uff08worst\uff09:   {ev._fmt_metrics(worst_loss, worst_p, worst_s)}")
        ev.log(f"  \u25b8 \u754c\u9650\uff08limit\uff09:   {ev._fmt_constraints(limit_loss, limit_p, limit_s)}")

        # baseline / worst \u7684\u6807\u51c6\u5dee\uff1a\u7528\u4e8e\u52a8\u6001 std \u4e0a\u754c\uff08\u4e0e limit \u540c quartile\uff09
        baseline_loss_std = float(baseline_reference_stats["loss_std"])
        baseline_m1_std = float(baseline_reference_stats["p_std"])
        baseline_m2_std = float(baseline_reference_stats["s_std"])
        worst_loss_std = float(worst_reference_stats["loss_std"])
        worst_m1_std = float(worst_reference_stats["p_std"])
        worst_m2_std = float(worst_reference_stats["s_std"])
        dynamic_loss_std_cap = _compute_dynamic_std_upper_bound(
            baseline_loss_std, worst_loss_std,
        )
        dynamic_m1_std_cap = _compute_dynamic_std_upper_bound(
            baseline_m1_std, worst_m1_std,
        )
        dynamic_m2_std_cap = _compute_dynamic_std_upper_bound(
            baseline_m2_std, worst_m2_std,
        )

        _noise_block_title(ev.log, "\u8bad\u7ec3\u8d85\u53c2\u4e0e\u8bbe\u5b9a\u6458\u8981\uff08Training hyperparameters V2\uff09")
        ev.log(
            f"  \u25b8 GTrXL: d_model={NOISE_STAGE_GTRXL_D_MODEL}, heads={NOISE_STAGE_GTRXL_N_HEADS}, "
            f"layers={NOISE_STAGE_GTRXL_N_LAYERS}, d_ff={NOISE_STAGE_GTRXL_D_FF}"
        )
        ev.log(
            f"  \u25b8 PPO: max_episodes={stage2_total_episodes}, eps_clip={NOISE_STAGE_PPO_EPS_CLIP}, "
            f"k_epochs={NOISE_STAGE_PPO_K_EPOCHS}, warmup={NOISE_STAGE_GTRXL_WARMUP_MODE}"
        )
        ev.log(
            f"  \u25b8 MC: train_samples={NOISE_STAGE_MC_TRAIN_SAMPLES}, confirm_samples={NOISE_STAGE_MC_CONFIRM_SEGMENTS}"
            f"  (baseline_segments={baseline_segments}, min_per_seg={NOISE_STAGE_MIN_SAMPLES_PER_SEGMENT})"
        )
        ev.log(
            f"  \u25b8 \u65b9\u5dee\u60e9\u7f5a: lambda={NOISE_STAGE_STABILITY_LAMBDA}  "
            f"\u00b7 \u52a8\u6001 loss_std \u4e0a\u754c={dynamic_loss_std_cap:.6f}  "
            f"(baseline {baseline_loss_std:.6f} \u2192 worst {worst_loss_std:.6f}, q={NOISE_STAGE_DYNAMIC_LIMIT_QUARTILE})"
        )
        ev.log(
            f"  \u25b8 \u52a8\u6001 m1_std \u4e0a\u754c={dynamic_m1_std_cap:.6f}  "
            f"(baseline {baseline_m1_std:.6f} \u2192 worst {worst_m1_std:.6f})  "
            f"\u00b7 m2_std \u4e0a\u754c={dynamic_m2_std_cap:.6f}  "
            f"({baseline_m2_std:.6f} \u2192 {worst_m2_std:.6f})"
        )
        ev.log(
            f"  \u25b8 \u5956\u52b1\u6743\u91cd: margin={NOISE_STAGE_TAIL_MARGIN_WEIGHT}, "
            f"violation={NOISE_STAGE_TAIL_VIOLATION_WEIGHT}, cost={NOISE_STAGE_COST_WEIGHT}"
        )
        ev.log(
            f"  \u25b8 \u57fa\u7ebf\u6807\u51c6\u5dee: loss_std={baseline_loss_std:.6f}, "
            f"m1_std={baseline_m1_std:.6f}, m2_std={baseline_m2_std:.6f}"
        )
        os.makedirs(noise_progress_dir, exist_ok=True)

        noise_step_info_details_dir = os.path.join(os.path.dirname(ev.noise_step_info_file), "details")
        os.makedirs(noise_step_info_details_dir, exist_ok=True)
        noise_step_info_chunk_file = [None]
        noise_step_info_chunk_idx = [0]
        noise_warning_file = os.path.join(os.path.dirname(ev.noise_step_info_file), "warning.txt")
        noise_prev_avg_reward = [None]
        noise_warnings = []

        def _get_noise_chunk_filename(episode_1based):
            chunk_start = ((episode_1based - 1) // STEP_INFO_CHUNK_SIZE) * STEP_INFO_CHUNK_SIZE + 1
            chunk_end = chunk_start + STEP_INFO_CHUNK_SIZE - 1
            return os.path.join(
                noise_step_info_details_dir,
                f"noise_ppo_step_info_{chunk_start}-{chunk_end}.txt",
            )

        def _open_noise_chunk(episode_1based):
            target = _get_noise_chunk_filename(episode_1based)
            new_idx = (episode_1based - 1) // STEP_INFO_CHUNK_SIZE
            if noise_step_info_chunk_file[0] is not None and noise_step_info_chunk_idx[0] == new_idx:
                return noise_step_info_chunk_file[0]
            if noise_step_info_chunk_file[0] is not None:
                noise_step_info_chunk_file[0].close()
            chunk_start = new_idx * STEP_INFO_CHUNK_SIZE + 1
            chunk_end = chunk_start + STEP_INFO_CHUNK_SIZE - 1
            f = open(target, "w", encoding="utf-8")
            f.write(
                f"=== \u566a\u58f0 PPO V2 \u9010\u6b65\u8bca\u65ad \u00b7 \u56de\u5408\u533a\u95f4 {chunk_start}-{chunk_end} ===\n\n"
            )
            noise_step_info_chunk_file[0] = f
            noise_step_info_chunk_idx[0] = new_idx
            return f

        # \u4fdd\u5b58\u539f\u59cb total_episodes\uff0c\u8bad\u7ec3\u7ed3\u675f\u540e\u6062\u590d
        original_total_episodes = getattr(ev, "total_episodes", stage2_total_episodes)
        ev.total_episodes = stage2_total_episodes
        ev._reset_runtime_ppo_state()

        # \u521b\u5efa\u7b56\u7565\u7f51\u7edc
        noise_net = _NoiseGTrXLStrategyNetwork(
            num_layers=ev.total_layers,
            d_model=NOISE_STAGE_GTRXL_D_MODEL,
            n_heads=NOISE_STAGE_GTRXL_N_HEADS,
            n_gtrxl_layers=NOISE_STAGE_GTRXL_N_LAYERS,
            d_ff=NOISE_STAGE_GTRXL_D_FF,
            dropout=NOISE_STAGE_GTRXL_DROPOUT,
            gtrxl_block_cls=GTrXLBlock,
            lstm_pos_dim=LSTM_POS_DIM,
            lstm_proj_dim=LSTM_PROJ_DIM,
            noise_stage_num_actions=NOISE_STAGE_NUM_ACTIONS,
            noise_stage_sos_tokens=NOISE_STAGE_SOS_TOKENS,
            noise_stage_prev_action_embed_dim=NOISE_STAGE_PREV_ACTION_EMBED_DIM,
            noise_stage_cont_dim=NOISE_STAGE_V2_CONT_DIM,
            noise_stage_action_dims=NOISE_STAGE_ACTION_DIMS,
        ).to(ev.device)
        optimizer = optim.Adam(noise_net.parameters(), lr=ev.ppo_lr_initial)
        noise_ppo_update_count = 0

        # \u8bc4\u4f30\u5668\u5305\u88c5\u5668
        class _NoiseRLEvaluatorWrapper:
            def __init__(wrapper_self, evaluator, fixed_gelu, fixed_softmax, split_name):
                wrapper_self.evaluator = evaluator
                wrapper_self.fixed_gelu = np.asarray(fixed_gelu, dtype=int)
                wrapper_self.fixed_softmax = np.asarray(fixed_softmax, dtype=int)
                wrapper_self.split_name = split_name

            def evaluate_noise_model(wrapper_self, **noise_kwargs):
                return wrapper_self.evaluator.evaluate_model_with_attention_noise(
                    wrapper_self.fixed_gelu,
                    wrapper_self.fixed_softmax,
                    split=wrapper_self.split_name,
                    **noise_kwargs,
                )

            def evaluate_noise_model_repeated(wrapper_self, repeats, **noise_kwargs):
                return wrapper_self.evaluator.evaluate_model_with_attention_noise_repeated(
                    wrapper_self.fixed_gelu,
                    wrapper_self.fixed_softmax,
                    repeats=repeats,
                    split=wrapper_self.split_name,
                    **noise_kwargs,
                )

            def evaluate_noise_model_segmented(wrapper_self, segments, **noise_kwargs):
                return wrapper_self.evaluator.evaluate_model_with_attention_noise_segmented(
                    wrapper_self.fixed_gelu,
                    wrapper_self.fixed_softmax,
                    segments=segments,
                    split=wrapper_self.split_name,
                    **noise_kwargs,
                )

        rl_evaluator = _NoiseRLEvaluatorWrapper(
            ev,
            fixed_gelu=fixed_gelu,
            fixed_softmax=fixed_softmax,
            split_name=reward_reference_split,
        )

        # \u521b\u5efa\u73af\u5883
        num_metrics = ev.get_num_metrics()
        env = _NoiseOptEnv(
            ev.total_layers,
            cost_reference_tot_c,
            (base_loss, base_p, base_s),
            rl_evaluator,
            fixed_gelu=fixed_gelu,
            fixed_softmax=fixed_softmax,
            constraint_limits=search_limits,
            num_metrics=num_metrics,
            input_noise_allowed=INPUT_NOISE_ALLOWED_SCALING_FACTORS,
            weight_noise_allowed=WEIGHT_NOISE_ALLOWED_SCALING_FACTORS,
            wffn1_noise_allowed=WFFN1_NOISE_ALLOWED_SCALING_FACTORS,
            input_noise_cost_map=INPUT_NOISE_COST_MAP,
            weight_noise_cost_map=WEIGHT_NOISE_COST_MAP,
            wffn1_noise_cost_map=WFFN1_NOISE_COST_MAP,
            input_noise_scaling_map=INPUT_NOISE_SCALING_MAP,
            wq_noise_scaling_map=WQ_NOISE_SCALING_MAP,
            wk_noise_scaling_map=WK_NOISE_SCALING_MAP,
            wv_noise_scaling_map=WV_NOISE_SCALING_MAP,
            wo_noise_scaling_map=WO_NOISE_SCALING_MAP,
            wffn1_noise_scaling_map=WFFN1_NOISE_SCALING_MAP,
            wffn2_noise_scaling_map=WFFN2_NOISE_SCALING_MAP,
            input_noise_scaling_to_norm=INPUT_NOISE_SCALING_TO_NORM,
            weight_noise_scaling_to_norm=WEIGHT_NOISE_SCALING_TO_NORM,
            wffn1_noise_scaling_to_norm=WFFN1_NOISE_SCALING_TO_NORM,
            noise_stage_sos_tokens=NOISE_STAGE_SOS_TOKENS,
            noise_stage_num_actions=NOISE_STAGE_NUM_ACTIONS,
            history_mask_value=HISTORY_MASK_VALUE,
            reward_clip_min=REWARD_CLIP_MIN,
            reward_clip_max=REWARD_CLIP_MAX,
            perf_weight_loss=NOISE_STAGE_PERF_WEIGHT_LOSS,
            perf_weight_m1=NOISE_STAGE_PERF_WEIGHT_M1,
            perf_weight_m2=NOISE_STAGE_PERF_WEIGHT_M2,
            barrier_weight_loss=NOISE_STAGE_BARRIER_WEIGHT_LOSS,
            barrier_weight_m1=NOISE_STAGE_BARRIER_WEIGHT_M1,
            barrier_weight_m2=NOISE_STAGE_BARRIER_WEIGHT_M2,
            mc_train_samples=_auto_adjust_segments(
                _eval_dataset_size, NOISE_STAGE_MC_TRAIN_SAMPLES, log_fn=ev.log,
            ),
            stability_lambda=NOISE_STAGE_STABILITY_LAMBDA,
            stability_base_std_tolerance=NOISE_STAGE_STABILITY_BASE_STD_TOLERANCE,
            baseline_loss_std=baseline_loss_std,
            baseline_m1_std=baseline_m1_std,
            dynamic_loss_std_cap=dynamic_loss_std_cap,
            dynamic_m1_std_cap=dynamic_m1_std_cap,
            dynamic_m2_std_cap=dynamic_m2_std_cap,
            noise_debt_log_alpha=NOISE_STAGE_NOISE_DEBT_LOG_ALPHA,
        )

        buffer = _NoiseRecurrentRolloutBuffer()
        noise_scaling_keys = tuple(
            key for key in cost_reference_noise_config.keys()
            if key.endswith("scaling_factors")
        )

        def _candidate_signature(candidate):
            return tuple(
                tuple(np.asarray(candidate[key], dtype=int).tolist())
                for key in noise_scaling_keys
            )

        def _clone_candidate(candidate):
            if candidate is None:
                return None
            cloned = {}
            for key, value in candidate.items():
                if isinstance(value, np.ndarray):
                    cloned[key] = value.copy()
                elif isinstance(value, dict):
                    cloned[key] = dict(value)
                elif isinstance(value, list):
                    cloned[key] = list(value)
                else:
                    cloned[key] = value
            return cloned

        # \u8ddf\u8e2a\u6570\u636e
        episode_returns = []
        episode_raw_final_rewards = []
        episode_final_selection_scores = []
        episode_losses = []
        episode_metric1s = []
        episode_metric2s = []
        episode_entropies = []

        best_final_selection_score = float("-inf")
        best_cost = float("inf")
        best_noise_config = None
        incumbent_best_noise_config = None
        incumbent_best_signature = None
        incumbent_mean_score = float("-inf")
        incumbent_history = []

        # \u521d\u59cb\u5b88\u64c2\u65b9\u6848\uff1a\u4f4e\u98ce\u9669\u57fa\u7ebf
        incumbent_cost_value, _ = ev.get_noise_simulated_cost(**baseline_noise_config)
        initial_incumbent_config = {
            key: np.asarray(value, dtype=int).copy()
            for key, value in baseline_noise_config.items()
        }
        initial_incumbent_config["cost"] = float(incumbent_cost_value)

        # \u521d\u59cb\u5b88\u64c2\u7edf\u8ba1\uff08\u5206\u6bb5\u8bc4\u6d4b\uff0c\u6574\u4efd\u9a8c\u8bc1\u96c6\u5207 N \u6bb5\uff0c\u53ea\u8dd1 1 \u904d\uff09
        best_test_segments = _auto_adjust_segments(
            _eval_dataset_size, NOISE_STAGE_BEST_TEST_SEGMENTS, log_fn=ev.log,
        )
        baseline_confirm_stats = ev.evaluate_model_with_attention_noise_segmented(
            fixed_gelu, fixed_softmax,
            **baseline_noise_config,
            segments=best_test_segments,
            split=reward_reference_split,
        )
        incumbent_best_noise_config = _clone_candidate(initial_incumbent_config)
        incumbent_best_noise_config["search_score_mean"] = float(baseline_confirm_stats["loss_mean"])
        incumbent_best_noise_config["search_score_std"] = float(baseline_confirm_stats["loss_std"])
        incumbent_best_noise_config["qualification_passed"] = True
        incumbent_best_noise_config["raw_final_reward"] = 0.0
        incumbent_best_noise_config["final_selection_score"] = 0.0
        incumbent_best_signature = _candidate_signature(initial_incumbent_config)
        incumbent_mean_score = 0.0  # \u57fa\u7ebf\u5206\u6570\u53c2\u8003
        incumbent_history.append({
            "source": "initial_incumbent",
            "episode": 0,
            "mean_score": 0.0,
            "cost": float(incumbent_cost_value),
        })
        ev.log(
            f"  [\u6700\u4f18\u590d\u8bd5 BestTest] \u521d\u59cb\u5b88\u64c2\u5df2\u5c31\u7eea\uff08\u57fa\u7ebf\u65e0\u6761\u4ef6\u4e0a\u53f0\uff09: "
            f"\u6210\u672c cost={incumbent_cost_value:.2f}"
        )

        # ============================
        # 断点续训：加载 checkpoint
        # ============================
        resume_start_episode = 0
        noise_rl_checkpoint_path = os.path.join(
            noise_progress_dir, NOISE_STAGE_CHECKPOINT_FILENAME,
        )
        if resume_checkpoint_path and os.path.isfile(resume_checkpoint_path):
            _noise_block_title(ev.log, "断点续训（Resume from checkpoint）")
            ev.log(f"  ▸ 加载 checkpoint: {resume_checkpoint_path}")
            ckpt = load_noise_rl_checkpoint(
                resume_checkpoint_path, noise_net, optimizer, device=ev.device,
            )
            resume_start_episode = int(ckpt["completed_episodes"])
            noise_ppo_update_count = int(ckpt["noise_ppo_update_count"])
            episode_returns = list(ckpt["episode_returns"])
            episode_raw_final_rewards = list(ckpt["episode_raw_final_rewards"])
            episode_final_selection_scores = list(ckpt["episode_final_selection_scores"])
            episode_losses = list(ckpt["episode_losses"])
            episode_metric1s = list(ckpt["episode_metric1s"])
            episode_metric2s = list(ckpt["episode_metric2s"])
            episode_entropies = list(ckpt["episode_entropies"])
            best_final_selection_score = float(ckpt["best_final_selection_score"])
            best_cost = float(ckpt["best_cost"])
            best_noise_config = _clone_candidate(ckpt["best_noise_config"])
            if ckpt.get("incumbent_best_noise_config") is not None:
                incumbent_best_noise_config = _clone_candidate(ckpt["incumbent_best_noise_config"])
            incumbent_best_signature = ckpt.get("incumbent_best_signature")
            incumbent_mean_score = float(ckpt.get("incumbent_mean_score", float("-inf")))
            incumbent_history = list(ckpt.get("incumbent_history", []))
            noise_prev_avg_reward[0] = ckpt.get("noise_prev_avg_reward")
            noise_warnings = list(ckpt.get("noise_warnings", []))
            _ev_rt = ckpt.get("ev_runtime_state", {})
            ev.reward_history = list(_ev_rt.get("reward_history", []))
            ev.reward_mean = float(_ev_rt.get("reward_mean", 0.0))
            ev.reward_std = float(_ev_rt.get("reward_std", 1.0))
            ev.current_episode = int(_ev_rt.get("current_episode", resume_start_episode))
            ev.log(
                f"  ▸ 已恢复至回合 {resume_start_episode}，"
                f"将从回合 {resume_start_episode + 1} 继续训练至 {stage2_total_episodes}"
            )
            if resume_start_episode >= stage2_total_episodes:
                ev.log(
                    f"  ⚠ checkpoint 已完成 {resume_start_episode} 回合，"
                    f"目标回合数 {stage2_total_episodes} 无需追加训练。"
                )

        # ============================
        # 主训练循环
        # ============================
        for episode in range(resume_start_episode, stage2_total_episodes):
            current_lr, current_entropy = ev.update_hyperparameters(optimizer, episode)
            env.set_episode_progress(episode, stage2_total_episodes)
            state = env.reset()
            prev_actions = torch.tensor(
                [list(NOISE_STAGE_SOS_TOKENS)], dtype=torch.long, device=ev.device
            ).unsqueeze(0)
            seq_cont_feats = []
            seq_layer_indices = []
            seq_prev_actions = []
            step_infos = []
            episode_reward_raw = 0.0
            episode_raw_final_reward = None
            episode_final_selection_score = None
            buffer.start_episode()

            for step in range(ev.total_layers):
                layer_idx = env.current_layer
                cont_feat_np = env.get_continuous_features()
                cont_feat = torch.tensor(
                    cont_feat_np, dtype=torch.float32, device=ev.device,
                ).view(1, 1, -1)
                layer_tensor = torch.tensor(
                    [[layer_idx]], dtype=torch.long, device=ev.device,
                )

                seq_cont_feats.append(cont_feat)
                seq_layer_indices.append(layer_tensor)
                seq_prev_actions.append(prev_actions)

                cont_seq = torch.cat(seq_cont_feats, dim=1)
                layer_seq = torch.cat(seq_layer_indices, dim=1)
                prev_action_seq = torch.cat(seq_prev_actions, dim=1)

                actions, logprob, value, prob_list = noise_net.get_action_and_logprob(
                    cont_seq, layer_seq, prev_action_seq, return_probs=True,
                )
                next_state, reward, done, info = env.step(
                    *[a.item() for a in actions]
                )

                if done:
                    episode_raw_final_reward = float(info.get("raw_final_reward", 0.0))
                    episode_final_selection_score = float(
                        info.get("final_selection_score", episode_raw_final_reward)
                    )

                step_info = {
                    "step_global": episode * ev.total_layers + step,
                    "episode_id": episode,
                    "layer_index": info["layer_index"],
                    "curr_input_noise_scaling_factor": info["curr_input_noise_scaling_factor"],
                    "curr_wq_noise_scaling_factor": info["curr_wq_noise_scaling_factor"],
                    "curr_wk_noise_scaling_factor": info["curr_wk_noise_scaling_factor"],
                    "curr_wv_noise_scaling_factor": info["curr_wv_noise_scaling_factor"],
                    "curr_wo_noise_scaling_factor": info["curr_wo_noise_scaling_factor"],
                    "curr_wffn1_noise_scaling_factor": info["curr_wffn1_noise_scaling_factor"],
                    "curr_wffn2_noise_scaling_factor": info["curr_wffn2_noise_scaling_factor"],
                    "x_prob_dist": prob_list[0].detach().cpu().numpy().tolist(),
                    "wq_prob_dist": prob_list[1].detach().cpu().numpy().tolist(),
                    "wk_prob_dist": prob_list[2].detach().cpu().numpy().tolist(),
                    "wv_prob_dist": prob_list[3].detach().cpu().numpy().tolist(),
                    "wo_prob_dist": prob_list[4].detach().cpu().numpy().tolist(),
                    "wffn1_prob_dist": prob_list[5].detach().cpu().numpy().tolist(),
                    "wffn2_prob_dist": prob_list[6].detach().cpu().numpy().tolist(),
                    "critic_value": value.item(),
                    "accumulated_cost": info["accumulated_cost"],
                    "accumulated_noise_variance": info["accumulated_noise_variance"],
                    "noise_debt_ratio": cont_feat_np[3],
                    "step_reward": reward,
                    "raw_final_reward": info.get("raw_final_reward"),
                    "final_selection_score": info.get("final_selection_score"),
                }
                mc_eval = info.get("mc_eval") or {}
                step_info["mc_loss_mean"] = mc_eval.get("loss_mean")
                step_info["mc_loss_std"] = mc_eval.get("loss_std")
                step_info["mc_metric1_mean"] = mc_eval.get("p_mean")
                step_info["mc_metric1_std"] = mc_eval.get("p_std")
                step_infos.append(step_info)

                buffer.add_step(
                    cont_feat=torch.tensor(cont_feat_np, dtype=torch.float32),
                    layer_idx=layer_idx,
                    prev_actions=prev_actions.squeeze(0).squeeze(0).detach().cpu(),
                    actions=actions.detach().cpu(),
                    logprob=logprob.detach().cpu(),
                    reward=reward,
                    value=value.detach().cpu(),
                    done=float(done),
                )

                prev_actions = actions.view(1, 1, -1).to(ev.device)
                episode_reward_raw += reward
                state = next_state

            buffer.end_episode()
            episode_returns.append(episode_reward_raw)
            episode_raw_final_rewards.append(
                episode_raw_final_reward if episode_raw_final_reward is not None else 0.0
            )
            if env.current_episode_metrics is not None:
                episode_losses.append(env.current_episode_metrics["loss"])
                episode_metric1s.append(env.current_episode_metrics["metric1"])
                episode_metric2s.append(env.current_episode_metrics["metric2"])
            else:
                episode_losses.append(base_loss)
                episode_metric1s.append(base_p)
                episode_metric2s.append(base_s)

            ev.update_reward_statistics(episode_reward_raw)

            # \u5199\u5165\u9010\u6b65\u8bca\u65ad
            chunk_f = _open_noise_chunk(episode + 1)
            chunk_f.write(
                f"\u2500\u2500 \u56de\u5408 episode {episode + 1} \u2500\u2500  "
                f"\u56de\u62a5 episode_return={episode_reward_raw:.4f}  ;  "
                f"\u539f\u59cb\u7ec8\u5c40 raw_final_reward="
                f"{(episode_raw_final_reward if episode_raw_final_reward is not None else 0.0):.4f}\n"
            )
            for si in step_infos:
                _write_noise_step_info(si, chunk_f)
                chunk_f.write("\n")
            chunk_f.flush()

            # \u6784\u5efa\u5f53\u524d episode \u7684 noise config
            final_noise_config = {
                "input_noise_scaling_factors": np.array(env.input_noise_config, dtype=int),
                "wq_noise_scaling_factors": np.array(env.wq_noise_config, dtype=int),
                "wk_noise_scaling_factors": np.array(env.wk_noise_config, dtype=int),
                "wv_noise_scaling_factors": np.array(env.wv_noise_config, dtype=int),
                "wo_noise_scaling_factors": np.array(env.wo_noise_config, dtype=int),
                "wffn1_noise_scaling_factors": np.array(env.wffn1_noise_config, dtype=int),
                "wffn2_noise_scaling_factors": np.array(env.wffn2_noise_config, dtype=int),
                "cost": env.accumulated_cost,
                "reward": episode_reward_raw,
                "raw_final_reward": episode_raw_final_reward if episode_raw_final_reward is not None else 0.0,
                "final_selection_score": (
                    episode_final_selection_score
                    if episode_final_selection_score is not None
                    else episode_reward_raw
                ),
                "episode_return": episode_reward_raw,
            }

            episode_final_selection_score_val = float(final_noise_config["final_selection_score"])
            episode_final_selection_scores.append(episode_final_selection_score_val)

            # \u66f4\u65b0\u5168\u5c40\u6700\u4f73
            if episode_final_selection_score_val > best_final_selection_score or (
                episode_final_selection_score_val == best_final_selection_score
                and env.accumulated_cost < best_cost
            ):
                best_final_selection_score = episode_final_selection_score_val
                best_cost = env.accumulated_cost
                best_noise_config = _clone_candidate(final_noise_config)
                _noise_block_title(ev.log, f"\u56de\u5408 {episode + 1} \u00b7 \u8bad\u7ec3\u8fc7\u7a0b\u65b0\u9ad8\uff08episode new best\uff09")
                ev.log(
                    f"  \u25b8 \u7ec8\u9009\u5206={episode_final_selection_score_val:.4f}  "
                    f"\u2502  \u6210\u672c cost={env.accumulated_cost:.2f}"
                )
                ev.log(
                    "  \u25b8 \u8bad\u7ec3\u671f\u6700\u4f18\u56de\u5408\u00b7\u566a\u58f0 scaling \u5e8f\u5217\uff08\u6309\u5c42\uff0c"
                    "\u4e0e Stage-2 \u52a8\u4f5c\u6620\u5c04\u540e\u7684 SF\uff09:"
                )
                for _bk in (
                    "input_noise_scaling_factors",
                    "wq_noise_scaling_factors",
                    "wk_noise_scaling_factors",
                    "wv_noise_scaling_factors",
                    "wo_noise_scaling_factors",
                    "wffn1_noise_scaling_factors",
                    "wffn2_noise_scaling_factors",
                ):
                    if _bk in best_noise_config:
                        ev.log(
                            f"     {_bk}: "
                            f"{np.asarray(best_noise_config[_bk], dtype=int).tolist()}"
                        )

            # \u6311\u6218\u8005\u786e\u8ba4\u673a\u5236
            challenger_signature = _candidate_signature(final_noise_config)
            if (
                (incumbent_best_signature is None or challenger_signature != incumbent_best_signature)
                and episode_final_selection_score_val > (
                    incumbent_mean_score + float(NOISE_STAGE_BEST_TEST_TRIGGER_MARGIN)
                )
            ):
                confirm_noise_kwargs = {
                    key: value.copy()
                    for key, value in final_noise_config.items()
                    if key.endswith("scaling_factors")
                }
                confirm_segments = _auto_adjust_segments(
                    _eval_dataset_size, NOISE_STAGE_MC_CONFIRM_SEGMENTS,
                )
                confirm_stats = ev.evaluate_model_with_attention_noise_segmented(
                    fixed_gelu, fixed_softmax,
                    segments=confirm_segments,
                    split=reward_reference_split,
                    **confirm_noise_kwargs,
                )
                confirm_loss_std = float(confirm_stats["loss_std"])
                confirm_m1_std = float(confirm_stats.get("p_std", 0.0))
                confirm_m2_std = float(confirm_stats.get("s_std", 0.0))

                # \u4e0e limit \u540c\u6784\u7684\u52a8\u6001 std \u4e0a\u754c\uff08baseline\u2192worst \u63d2\u503c + \u53cd\u5e38\u4e0d\u5916\u63a8\uff09
                std_check_pass = (
                    confirm_loss_std <= dynamic_loss_std_cap + 1e-8
                    and confirm_m1_std <= dynamic_m1_std_cap + 1e-8
                )
                if num_metrics > 1:
                    std_check_pass = std_check_pass and (
                        confirm_m2_std <= dynamic_m2_std_cap + 1e-8
                    )
                # \u68c0\u67e5\u7ea6\u675f\u6761\u4ef6
                constraint_pass = ev._candidate_meets_constraints(
                    confirm_stats["loss_mean"],
                    confirm_stats["p_mean"],
                    confirm_stats["s_mean"],
                    search_limits["loss"],
                    search_limits["metric1"],
                    search_limits["metric2"],
                )

                passed = std_check_pass and constraint_pass
                confirm_mean_score = episode_final_selection_score_val

                _std_lines = [
                    f"  \u65b9\u5dee\u68c0\u67e5 std_check={std_check_pass}  "
                    f"(loss_std {confirm_loss_std:.6f} vs cap {dynamic_loss_std_cap:.6f})",
                    f"  m1_std {confirm_m1_std:.6f} vs cap {dynamic_m1_std_cap:.6f}",
                ]
                if num_metrics > 1:
                    _std_lines.append(
                        f"  m2_std {confirm_m2_std:.6f} vs cap {dynamic_m2_std_cap:.6f}"
                    )
                _log_rounded_box(ev.log, [
                    f"\u6311\u6218\u8005\u786e\u8ba4\uff08Challenger Confirmation\uff09\u00b7 \u56de\u5408 {episode + 1}",
                    f"  N={confirm_segments} \u5206\u6bb5\u8bc4\u6d4b",
                    f"  loss={confirm_stats['loss_mean']:.4f}+/-{confirm_loss_std:.4f}, "
                    f"m1={confirm_stats['p_mean']:.4f}+/-{confirm_m1_std:.4f}",
                    *_std_lines,
                    f"  \u7ea6\u675f\u68c0\u67e5 constraint={constraint_pass}",
                    f"  \u88c1\u5b9a\uff08verdict\uff09: {'PASS' if passed else 'FAIL'}",
                ])

                if passed:
                    if confirm_mean_score > incumbent_mean_score:
                        incumbent_best_noise_config = _clone_candidate(final_noise_config)
                        incumbent_best_noise_config["qualification_passed"] = True
                        incumbent_best_noise_config["confirm_stats"] = _copy_repeat_summary(confirm_stats)
                        incumbent_best_signature = challenger_signature
                        incumbent_mean_score = confirm_mean_score
                        incumbent_history.append({
                            "source": "challenger_update",
                            "episode": int(episode) + 1,
                            "mean_score": float(confirm_mean_score),
                            "cost": float(final_noise_config["cost"]),
                        })
                        ev.log(
                            f"  \u2605 \u5b88\u64c2\u6700\u4f18\uff08incumbent best\uff09\u5df2\u66f4\u65b0 \u00b7 \u56de\u5408 {episode + 1}: "
                            f"\u65b0 mean_score={confirm_mean_score:.4f}, "
                            f"\u6210\u672c cost={final_noise_config['cost']:.2f}"
                        )
                else:
                    # \u786e\u8ba4\u5931\u8d25\uff1a\u5199\u56de\u60e9\u7f5a\u5230 buffer
                    if buffer.episodes:
                        buffer.episodes[-1]["rewards"][-1] = float(NOISE_STAGE_CONFIRM_FAIL_PENALTY)
                        ev.log(
                            f"  \u2716 \u786e\u8ba4\u5931\u8d25\uff0c\u5df2\u5c06\u7ec8\u7aef\u5956\u52b1\u8986\u76d6\u4e3a "
                            f"CONFIRM_FAIL_PENALTY={NOISE_STAGE_CONFIRM_FAIL_PENALTY}"
                        )

            # PPO \u66f4\u65b0
            if (episode + 1) % PPO_UPDATE_INTERVAL == 0:
                policy_loss, value_loss, entropy = _ppo_update_noise_gtrxl(
                    ev, noise_net, optimizer, buffer, ev.device,
                    entropy_coef=current_entropy,
                    ppo_update_step=noise_ppo_update_count,
                    ppo_eps_clip=NOISE_STAGE_PPO_EPS_CLIP,
                    ppo_k_epochs=NOISE_STAGE_PPO_K_EPOCHS,
                    ppo_value_coef=PPO_VALUE_COEF,
                    gtrxl_warmup_mode=NOISE_STAGE_GTRXL_WARMUP_MODE,
                    gtrxl_warmup_updates=NOISE_STAGE_GTRXL_WARMUP_UPDATES,
                    gtrxl_short_warmup_updates=NOISE_STAGE_GTRXL_SHORT_WARMUP_UPDATES,
                    gtrxl_entropy_lower_bound=GTRXL_ENTROPY_LOWER_BOUND,
                    gtrxl_mini_batch_episodes=GTRXL_MINI_BATCH_EPISODES,
                            value_clip_range=NOISE_STAGE_VALUE_CLIP_RANGE,
                )
                noise_ppo_update_count += 1
                buffer.clear()
                episode_entropies.append(entropy)
                avg_episode_return = np.mean(episode_returns[-PPO_UPDATE_INTERVAL:])
                avg_raw_final_reward = np.mean(episode_raw_final_rewards[-PPO_UPDATE_INTERVAL:])
                warmup_status = "constant"
                if NOISE_STAGE_GTRXL_WARMUP_MODE != "constant":
                    warmup_status = (
                        "warmup"
                        if noise_ppo_update_count <= NOISE_STAGE_GTRXL_WARMUP_UPDATES
                        else "normal"
                    )
                _log_rounded_box(
                    ev.log,
                    [
                        f"PPO \u7a97\u53e3\u6458\u8981 \u00b7 \u622a\u81f3\u56de\u5408\uff08through episode\uff09 {episode + 1}",
                        f"\u5e73\u5747\u56de\u62a5 mean return={avg_episode_return:.4f}  ;  "
                        f"\u5e73\u5747\u7ec8\u5c40\u5956\u52b1 mean final={avg_raw_final_reward:.4f}",
                        f"\u7b56\u7565\u635f\u5931 policy_loss={policy_loss:.4f}  ;  "
                        f"\u4ef7\u503c\u635f\u5931 value_loss={value_loss:.4f}  ;  \u71b5 entropy={entropy:.4f}",
                        f"LR={optimizer.param_groups[0]['lr']:.6f}, "
                        f"\u71b5\u7cfb\u6570 entropy_coef={current_entropy:.6f}, "
                        f"\u66f4\u65b0\u5e8f\u53f7 update#{noise_ppo_update_count}, "
                        f"\u72b6\u6001={_warmup_status_zh(warmup_status)}",
                    ],
                )

                # \u5956\u52b1\u9a9f\u964d\u68c0\u6d4b
                if noise_prev_avg_reward[0] is not None:
                    reward_drop = noise_prev_avg_reward[0] - avg_raw_final_reward
                    if reward_drop > REWARD_DROP_WARNING_THRESHOLD:
                        window_start_ep = episode + 1 - PPO_UPDATE_INTERVAL + 1
                        window_end_ep = episode + 1
                        involved_files = sorted(set(
                            _get_noise_chunk_filename(e)
                            for e in range(window_start_ep, window_end_ep + 1)
                        ))
                        involved_basenames = [os.path.basename(fp) for fp in involved_files]
                        warn_msg = {
                            "type": "\u566a\u58f0\u9636\u6bb5\u5956\u52b1\u9a9f\u964d",
                            "window": noise_ppo_update_count,
                            "prev_avg": float(noise_prev_avg_reward[0]),
                            "curr_avg": float(avg_raw_final_reward),
                            "drop": float(reward_drop),
                            "threshold": float(REWARD_DROP_WARNING_THRESHOLD),
                            "episode_range": (window_start_ep, window_end_ep),
                            "detail_files": involved_basenames,
                        }
                        noise_warnings.append(warn_msg)
                        _log_rounded_box(ev.log, [
                            f"\u26a0 \u5956\u52b1\u9a9f\u964d\u8b66\u544a \u00b7 \u7b2c {len(noise_warnings)} \u6761",
                            f"\u7a97\u53e3: {noise_ppo_update_count}, \u56de\u5408: {window_start_ep}-{window_end_ep}",
                            f"\u4e0a\u6b21={noise_prev_avg_reward[0]:.4f} -> \u672c\u6b21={avg_raw_final_reward:.4f}, "
                            f"\u964d\u5e45={reward_drop:.4f}",
                        ])
                        _write_warning_report(noise_warning_file, noise_warnings,
                                              stage_label="\u566a\u58f0\u9636\u6bb5 V2\uff08Stage-2 Noise V2\uff09")
                noise_prev_avg_reward[0] = avg_raw_final_reward

            # \u8fdb\u5ea6\u5feb\u7167
            if (episode + 1) % NOISE_STAGE_PROGRESS_SAVE_INTERVAL == 0:
                progress_training_curve_path = os.path.join(
                    noise_progress_dir,
                    f"noise_ppo_training_curve_ep{episode + 1}.png",
                )
                progress_entropy_curve_path = os.path.join(
                    noise_progress_dir,
                    f"noise_ppo_entropy_curve_ep{episode + 1}.png",
                )
                _plot_noise_training_curves(
                    ev,
                    episode_returns, episode_raw_final_rewards,
                    episode_losses, episode_metric1s, episode_metric2s,
                    episode_entropies,
                    base_loss=base_loss, base_p=base_p, base_s=base_s,
                    training_curve_path=progress_training_curve_path,
                    entropy_curve_path=progress_entropy_curve_path,
                    ppo_update_interval=PPO_UPDATE_INTERVAL,
                )
                ev.log(
                    f"  [\u8fdb\u5ea6] \u5df2\u5199\u5165\u8bad\u7ec3\u66f2\u7ebf\u5feb\u7167 \u00b7 \u56de\u5408 {episode + 1}"
                )
                # 保存 checkpoint（断点续训用）
                save_noise_rl_checkpoint(
                    path=noise_rl_checkpoint_path,
                    noise_net=noise_net,
                    optimizer=optimizer,
                    episode=episode,
                    noise_ppo_update_count=noise_ppo_update_count,
                    episode_returns=episode_returns,
                    episode_raw_final_rewards=episode_raw_final_rewards,
                    episode_final_selection_scores=episode_final_selection_scores,
                    episode_losses=episode_losses,
                    episode_metric1s=episode_metric1s,
                    episode_metric2s=episode_metric2s,
                    episode_entropies=episode_entropies,
                    best_final_selection_score=best_final_selection_score,
                    best_cost=best_cost,
                    best_noise_config=best_noise_config,
                    incumbent_best_noise_config=incumbent_best_noise_config,
                    incumbent_best_signature=incumbent_best_signature,
                    incumbent_mean_score=incumbent_mean_score,
                    incumbent_history=incumbent_history,
                    ev_runtime_state={
                        "reward_history": list(ev.reward_history),
                        "reward_mean": float(ev.reward_mean),
                        "reward_std": float(ev.reward_std),
                        "current_episode": int(ev.current_episode),
                    },
                    noise_prev_avg_reward=noise_prev_avg_reward[0],
                    noise_warnings=noise_warnings,
                )
                ev.log(
                    f"  [进度] 已保存 checkpoint · 回合 {episode + 1} → {noise_rl_checkpoint_path}"
                )

        # ============================
        # \u8bad\u7ec3\u7ed3\u675f
        # ============================
        status = NOISE_STAGE_STATUS_OK
        selected_config = (
            _clone_candidate(incumbent_best_noise_config)
            if incumbent_best_noise_config is not None
            and incumbent_best_noise_config.get("qualification_passed", False)
            else None
        )
        if selected_config is None:
            status = NOISE_STAGE_STATUS_NO_STABLE_FEASIBLE
            ev.log(
                "\n  [\u7ed3\u679c] \u672a\u627e\u5230\u7a33\u5b9a\u53ef\u884c\u7684\u566a\u58f0\u65b9\u6848\u3002"
            )

        _noise_log_major_rule(ev.log, "\u4e8c\u9636\u6bb5\u566a\u58f0 PPO V2 \u8bad\u7ec3\u7ed3\u675f\uff08Noise PPO V2 training completed\uff09")
        ev.log(
            f"  \u25b8 \u9636\u6bb5\u72b6\u6001 stage status: "
            f"{_noise_stage_status_zh(status, NOISE_STAGE_STATUS_OK, NOISE_STAGE_STATUS_NO_STABLE_FEASIBLE)}"
        )
        if selected_config is not None:
            _noise_block_title(ev.log, "\u9009\u5b9a\u914d\u7f6e\u8f93\u51fa\uff08Best stable noise configuration\uff09")
            for key in (
                "input_noise_scaling_factors", "wq_noise_scaling_factors",
                "wk_noise_scaling_factors", "wv_noise_scaling_factors",
                "wo_noise_scaling_factors", "wffn1_noise_scaling_factors",
                "wffn2_noise_scaling_factors",
            ):
                if key in selected_config:
                    ev.log(f"     {key}: {selected_config[key].tolist()}")
            ev.log(
                f"  \u25b8 \u6210\u672c cost={selected_config['cost']:.2f}  "
                f"\u2502 \u7ec8\u9009\u5206 final_selection_score={selected_config['final_selection_score']:.4f}"
            )

        # \u7ed8\u5236\u6700\u7ec8\u66f2\u7ebf
        _plot_noise_training_curves(
            ev, episode_returns, episode_raw_final_rewards,
            episode_losses, episode_metric1s, episode_metric2s, episode_entropies,
            base_loss=base_loss, base_p=base_p, base_s=base_s,
            training_curve_path=noise_training_curve_path,
            entropy_curve_path=noise_entropy_curve_path,
            ppo_update_interval=PPO_UPDATE_INTERVAL,
        )

        if noise_step_info_chunk_file[0] is not None:
            noise_step_info_chunk_file[0].close()
            noise_step_info_chunk_file[0] = None

        if noise_warnings:
            ev.log(
                f"  \u26a0 \u8bad\u7ec3\u671f\u95f4\u5171\u89e6\u53d1 {len(noise_warnings)} \u6761\u5956\u52b1\u9a9f\u964d\u544a\u8b66"
            )

        # 训练结束保存最终 checkpoint（断点续训用）
        _final_ep = stage2_total_episodes - 1
        if stage2_total_episodes > resume_start_episode:
            save_noise_rl_checkpoint(
                path=noise_rl_checkpoint_path,
                noise_net=noise_net,
                optimizer=optimizer,
                episode=_final_ep,
                noise_ppo_update_count=noise_ppo_update_count,
                episode_returns=episode_returns,
                episode_raw_final_rewards=episode_raw_final_rewards,
                episode_final_selection_scores=episode_final_selection_scores,
                episode_losses=episode_losses,
                episode_metric1s=episode_metric1s,
                episode_metric2s=episode_metric2s,
                episode_entropies=episode_entropies,
                best_final_selection_score=best_final_selection_score,
                best_cost=best_cost,
                best_noise_config=best_noise_config,
                incumbent_best_noise_config=incumbent_best_noise_config,
                incumbent_best_signature=incumbent_best_signature,
                incumbent_mean_score=incumbent_mean_score,
                incumbent_history=incumbent_history,
                ev_runtime_state={
                    "reward_history": list(ev.reward_history),
                    "reward_mean": float(ev.reward_mean),
                    "reward_std": float(ev.reward_std),
                    "current_episode": int(ev.current_episode),
                },
                noise_prev_avg_reward=noise_prev_avg_reward[0],
                noise_warnings=noise_warnings,
            )
            ev.log(f"  [完成] 最终 checkpoint 已保存 → {noise_rl_checkpoint_path}")

        ev.total_episodes = original_total_episodes
        ev.apply_configuration(fixed_gelu, fixed_softmax)
        ev.clear_input_noise_configuration()
        ev.clear_weight_noise_configuration()

        # 返回结果字典（兼容下游 noise_final_evaluation_module）
        return {
            "fixed_gelu": fixed_gelu.copy(),
            "fixed_softmax": fixed_softmax.copy(),
            "baseline_noise_config": {k: v.copy() for k, v in cost_reference_noise_config.items()},
            "baseline_tot_c": float(cost_reference_tot_c),
            "cost_reference_noise_config": {k: v.copy() for k, v in cost_reference_noise_config.items()},
            "cost_reference_source": "max_noise_configuration",
            "performance_baseline_gelu": fixed_gelu.copy(),
            "performance_baseline_softmax": fixed_softmax.copy(),
            "performance_baseline_source": "stage1_fixed_low_risk_noise",
            "baseline_segments": int(baseline_segments),
            "search_baseline_stats": _copy_repeat_summary(baseline_reference_stats),
            "worst_reference_stats": _copy_repeat_summary(worst_reference_stats),
            "worst_case_noise_config": {k: v.copy() for k, v in worst_case_noise_config.items()},
            "limit_computation_method": "dynamic_quartile",
            "limit_quartile": float(NOISE_STAGE_DYNAMIC_LIMIT_QUARTILE),
            "search_limits": {k: float(v) for k, v in search_limits.items()},
            "status": status,
            "best_noise_config": _clone_candidate(selected_config),
            "stable_search_best_noise_config": _clone_candidate(selected_config),
            "stable_joint_best_noise_config": _clone_candidate(selected_config),
            "selection_diagnostics": {
                "selection_mode": "incumbent_best_test_v2",
                "mc_train_samples": int(NOISE_STAGE_MC_TRAIN_SAMPLES),
                "mc_confirm_segments": int(NOISE_STAGE_MC_CONFIRM_SEGMENTS),
                "stability_lambda": float(NOISE_STAGE_STABILITY_LAMBDA),
                "confirm_fail_penalty": float(NOISE_STAGE_CONFIRM_FAIL_PENALTY),
                "noise_debt_log_alpha": float(NOISE_STAGE_NOISE_DEBT_LOG_ALPHA),
                "noise_debt_ratio_mode": "log2_minmax_total",
                "ppo_gamma": float(NOISE_STAGE_PPO_GAMMA),
                "value_clip_range": float(NOISE_STAGE_VALUE_CLIP_RANGE),
                "trigger_margin": float(NOISE_STAGE_BEST_TEST_TRIGGER_MARGIN),
                "dynamic_loss_std_cap": float(dynamic_loss_std_cap),
                "dynamic_m1_std_cap": float(dynamic_m1_std_cap),
                "dynamic_m2_std_cap": float(dynamic_m2_std_cap),
                "incumbent_history": [dict(item) for item in incumbent_history],
                "final_incumbent": _clone_candidate(incumbent_best_noise_config),
            },
            "shortlist_diagnostics": {},
            "limit_loss": float(limit_loss),
            "limit_p": float(limit_p),
            "limit_s": float(limit_s),
            "proxy_limit_loss": float(search_limits["loss"]),
            "proxy_limit_p": float(search_limits["metric1"]),
            "proxy_limit_s": float(search_limits["metric2"]),
            "proxy_base_loss": float(base_loss),
            "proxy_base_p": float(base_p),
            "proxy_base_s": float(base_s),
            "training_eval_split": str(reward_reference_split),
            "training_hparams": {
                "gtrxl_d_model": NOISE_STAGE_GTRXL_D_MODEL,
                "gtrxl_n_heads": NOISE_STAGE_GTRXL_N_HEADS,
                "gtrxl_n_layers": NOISE_STAGE_GTRXL_N_LAYERS,
                "gtrxl_d_ff": NOISE_STAGE_GTRXL_D_FF,
                "ppo_max_episodes": stage2_total_episodes,
                "ppo_update_interval": PPO_UPDATE_INTERVAL,
                "ppo_eps_clip": NOISE_STAGE_PPO_EPS_CLIP,
                "ppo_k_epochs": NOISE_STAGE_PPO_K_EPOCHS,
                "mc_train_samples": NOISE_STAGE_MC_TRAIN_SAMPLES,
                "mc_confirm_segments": NOISE_STAGE_MC_CONFIRM_SEGMENTS,
                "stability_lambda": NOISE_STAGE_STABILITY_LAMBDA,
                "stability_base_std_tolerance": NOISE_STAGE_STABILITY_BASE_STD_TOLERANCE,
                "dynamic_loss_std_cap": float(dynamic_loss_std_cap),
                "dynamic_m1_std_cap": float(dynamic_m1_std_cap),
                "dynamic_m2_std_cap": float(dynamic_m2_std_cap),
                "cost_weight": NOISE_STAGE_COST_WEIGHT,
                "tail_margin_weight": NOISE_STAGE_TAIL_MARGIN_WEIGHT,
                "tail_violation_weight": NOISE_STAGE_TAIL_VIOLATION_WEIGHT,
                "noise_debt_log_alpha": NOISE_STAGE_NOISE_DEBT_LOG_ALPHA,
                "ppo_gamma": float(NOISE_STAGE_PPO_GAMMA),
                "value_clip_range": float(NOISE_STAGE_VALUE_CLIP_RANGE),
                "v2_cont_dim": NOISE_STAGE_V2_CONT_DIM,
            },
            "reward_diagnostics": {
                "terminal_reward_mode": "mc_sparse_v2",
                "mc_train_samples": NOISE_STAGE_MC_TRAIN_SAMPLES,
                "episode_return_mean": float(np.mean(episode_returns)) if episode_returns else None,
                "final_selection_score_mean": float(np.mean(episode_final_selection_scores)) if episode_final_selection_scores else None,
                "raw_final_reward_mean": float(np.mean(episode_raw_final_rewards)) if episode_raw_final_rewards else None,
            },
        }
