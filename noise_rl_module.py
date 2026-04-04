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


# ---------------------------------------------------------------------------
# Stage-2 超参数（尾部安全 tail-safe 重构版）
# ---------------------------------------------------------------------------

NOISE_STAGE_GTRXL_D_MODEL = 256
NOISE_STAGE_GTRXL_N_HEADS = 8
NOISE_STAGE_GTRXL_N_LAYERS = 4
NOISE_STAGE_GTRXL_D_FF = 512
NOISE_STAGE_GTRXL_DROPOUT = 0.1

NOISE_STAGE_PPO_MAX_EPISODES = 40000
NOISE_STAGE_PPO_EPS_CLIP = 0.12
NOISE_STAGE_PPO_K_EPOCHS = 6
NOISE_STAGE_GTRXL_WARMUP_MODE = "constant"
NOISE_STAGE_GTRXL_WARMUP_UPDATES = 0
NOISE_STAGE_GTRXL_SHORT_WARMUP_UPDATES = 20

# ---------------------------------------------------------------------------
# 训练期 / 评估期 评估次数分离（文档修改方案 第1步）
# ---------------------------------------------------------------------------
# 训练期：动作选择相关，只做 1 次评估
NOISE_STAGE_TRAIN_DENSE_EVAL_REPEATS = 1       # 中间步 dense probe 评估次数
NOISE_STAGE_TRAIN_TERMINAL_EVAL_REPEATS = 1     # 终止步 terminal 评估次数
# 评估期：baseline / confirm / finalist 仍做多次评估
NOISE_STAGE_BASELINE_REPEATS = 50
NOISE_STAGE_ONLINE_BASELINE_REPEATS = 20
NOISE_STAGE_CONFIRM_REPEATS = 16
NOISE_STAGE_FINALIST_REPEATS = 64
NOISE_STAGE_BEST_TEST_REPEATS = 30
NOISE_STAGE_BEST_TEST_TRIGGER_MARGIN = 0.0
NOISE_STAGE_BEST_TEST_SAFE_RATE_MIN = 0.93
NOISE_STAGE_BEST_TEST_CVAR_MAX = 0.05
NOISE_STAGE_BEST_TEST_MAX_ENTRY_SCORE_DROP = 0.015
NOISE_STAGE_BEST_TEST_ENTRY_STD_MULTIPLIER = 2.0
NOISE_STAGE_SHORTLIST_SIZE = 10                 # 稍微增大（原8），降低单次训练 reward 偶然性影响

# 训练期 reward 模式配置（文档修改方案 第1步）
NOISE_STAGE_DENSE_REWARD_MODE = "prefix_completion_delta"       # 中间步 dense reward 模式
NOISE_STAGE_TRAIN_TERMINAL_REWARD_MODE = "single_shot_margin"   # 终止步训练期 reward 模式
NOISE_STAGE_DENSE_COMPLETION_POLICY = "max_suffix"              # 未来层补全策略

# 旧兼容项（不再用于训练期，仅保留给 confirm/finalist）
NOISE_STAGE_MC_SAMPLES = 5
NOISE_STAGE_MC_BASE_SAMPLES = 5
NOISE_STAGE_MC_EXTRA_SAMPLES = 0
NOISE_STAGE_MC_MARGIN_THRESHOLD = 0.02
NOISE_STAGE_BUDGET_DECAY_FRACTION = 0.5
NOISE_STAGE_PROGRESS_SAVE_INTERVAL = 10000
NOISE_STAGE_PROGRESS_DIR = os.path.join("rl_results", "noise_rl_progress")

# 二阶段「搜索 / 留出 / 训练 proxy」共用：由低风险 baseline 与最强噪声 worst-case 两点的指标线性插值得到约束界限 limit。
# 公式：limit = baseline + NOISE_STAGE_DYNAMIC_LIMIT_QUARTILE * (worst - baseline)。
# - Loss：worst 通常更高；该系数越小，limit 越靠近 baseline（损失上限越低 → 更严）；越大越靠近 worst（越松）。
# - 指标（如 Acc / F1）：worst 通常更低；系数越小，limit 越靠近 baseline（准确率下限等越高 → 更严）；越大越松。
# 建议取值区间 [0, 1]；0 等价于完全贴合 baseline，1 等价于完全贴合 worst。
NOISE_STAGE_DYNAMIC_LIMIT_QUARTILE = 0.2

# 兼容项：旧均值 reward 权重（不再主导，仅用于辅助/兼容）
NOISE_STAGE_FINAL_REWARD_ALPHA_PERF = 0.75
NOISE_STAGE_FINAL_REWARD_ALPHA_COST = 0.25
NOISE_STAGE_PERF_WEIGHT_LOSS = 0.15
NOISE_STAGE_PERF_WEIGHT_M1 = 0.425
NOISE_STAGE_PERF_WEIGHT_M2 = 0.425
NOISE_STAGE_BARRIER_WEIGHT_LOSS = 0.10
NOISE_STAGE_BARRIER_WEIGHT_M1 = 0.45
NOISE_STAGE_BARRIER_WEIGHT_M2 = 0.45
NOISE_STAGE_DENSE_REWARD_SHAPING_SCALE = 0.25

NOISE_STAGE_STABILITY_PROXY_STD_REF = 0.008
NOISE_STAGE_STABILITY_PENALTY_SCALE = 0.10
NOISE_STAGE_WEIGHT_TOL = 1e-6
NOISE_STAGE_STATUS_OK = "ok"
NOISE_STAGE_STATUS_NO_STABLE_FEASIBLE = "no_stable_feasible_candidate"
NOISE_STAGE_STABILITY_THRESHOLDS = {
    "search": {
        "loss_std": 0.008,
        "metric1_std": 0.008,
        "metric2_std": 0.008,
        "loss_sem_ratio": 0.00375,
        "metric1_sem_ratio": 0.0025,
        "metric2_sem_ratio": 0.0025,
    },
    "holdout": {
        "loss_std": 0.010,
        "metric1_std": 0.012,
        "metric2_std": 0.012,
        "loss_sem_ratio": 0.0050,
        "metric1_sem_ratio": 0.00375,
        "metric2_sem_ratio": 0.00375,
    },
}

# ---------------------------------------------------------------------------
# 尾部安全（tail-safe）相关超参数
# ---------------------------------------------------------------------------

# 尾部风险评估参数
NOISE_STAGE_TAIL_ALPHA_EVAL = 0.05
NOISE_STAGE_TAIL_TRAIN_K_MIN = 2
NOISE_STAGE_TAIL_CONFIRM_K_MIN = 2
NOISE_STAGE_TAIL_FINAL_K = 4  # math.ceil(0.05 * 64)

# 训练期 tail surrogate reward 权重
NOISE_STAGE_TAIL_MARGIN_WEIGHT = 1.00
NOISE_STAGE_TAIL_VIOLATION_WEIGHT = 1.25
NOISE_STAGE_SAFE_RATE_GAP_WEIGHT = 0.50
NOISE_STAGE_MEAN_PERF_WEIGHT = 0.20
NOISE_STAGE_COST_WEIGHT = 0.05

# 训练期安全率目标
NOISE_STAGE_TRAIN_SAFE_RATE_TARGET = 0.80

# confirm 阈值（16 次重复）
NOISE_STAGE_CONFIRM_SAFE_RATE_MIN = 0.875
NOISE_STAGE_CONFIRM_CVAR_MAX = 0.35

# finalist 阈值（64 次重复）
NOISE_STAGE_FINALIST_SAFE_RATE_MIN = 0.95
NOISE_STAGE_FINALIST_CVAR_MAX = 0.20
NOISE_STAGE_FINALIST_SAFE_RATE_LB95_MIN = 0.90

# 窗口候选 top-k（稍微增大，降低单次训练 reward 偶然性影响）
NOISE_STAGE_WINDOW_TOPK = 5

# 双头 critic：mean auxiliary head 的 loss 权重
NOISE_STAGE_MEAN_AUX_LOSS_COEF = 0.25

# Bootstrap 重采样次数（用于 finalist CVaR UCB）
NOISE_STAGE_BOOTSTRAP_ITERATIONS = 200


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
    base_loss,
    base_p,
    base_s,
    worst_loss,
    worst_p,
    worst_s,
    quartile=NOISE_STAGE_DYNAMIC_LIMIT_QUARTILE,
):
    """根据 baseline 和 worst-case 指标动态计算约束上下限。

    常规：limit = baseline + quartile * (worst - baseline)（在 baseline 与 worst 之间插值）。
    Loss 越小越好；metric1/metric2 越大越好。

    特殊情况（统计噪声或评估反常时，插值方向会使 limit 比 baseline 更松）：
    - 若 baseline_loss > worst_loss：不插值，limit_loss = baseline_loss。
    - 若 baseline_m1 < worst_m1：limit_metric1 = baseline_m1；m2 同理。
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


def _compute_tail_metrics_from_trials(
    trials, constraint_limits, baseline_metrics, perf_weights, barrier_weights,
    num_metrics, alpha=NOISE_STAGE_TAIL_ALPHA_EVAL, k_min=NOISE_STAGE_TAIL_TRAIN_K_MIN,
):
    """从 trial 级数组计算尾部风险指标（violation cost, margin utility, safe_rate 等）。

    适用于训练期 MC 评估和 confirm/finalist 阶段的重复评估。
    """
    baseline_loss, baseline_metric1, baseline_metric2 = baseline_metrics
    loss_limit = float(constraint_limits["loss"])
    m1_limit = float(constraint_limits["metric1"])
    m2_limit = float(constraint_limits.get("metric2", m1_limit)) if num_metrics > 1 else m1_limit

    n = len(trials)
    tail_k = max(k_min, math.ceil(alpha * n))

    per_trial_margins_loss = []
    per_trial_margins_m1 = []
    per_trial_margins_m2 = []
    per_trial_violation_costs = []
    per_trial_margin_utilities = []
    per_trial_safe = []

    for trial in trials:
        t_loss = float(trial["loss"])
        t_m1 = float(trial["metric1"])
        t_m2 = float(trial.get("metric2", t_m1)) if num_metrics > 1 else t_m1

        # 归一化 margin：正值=安全，负值=违约
        # 使用 abs() 防止 limit 在 baseline 同侧（如 worst_loss < baseline_loss）时
        # 分母为极小正数导致 margin 爆炸至数百万级别
        _MARGIN_DENOM_FLOOR = 0.01
        margin_loss = (loss_limit - t_loss) / max(abs(loss_limit - float(baseline_loss)), _MARGIN_DENOM_FLOOR)
        margin_m1 = (t_m1 - m1_limit) / max(abs(float(baseline_metric1) - m1_limit), _MARGIN_DENOM_FLOOR)
        if num_metrics > 1:
            margin_m2 = (t_m2 - m2_limit) / max(abs(float(baseline_metric2) - m2_limit), _MARGIN_DENOM_FLOOR)
        else:
            margin_m2 = 0.0

        per_trial_margins_loss.append(margin_loss)
        per_trial_margins_m1.append(margin_m1)
        per_trial_margins_m2.append(margin_m2)

        # 单次试验 violation cost（加权违约量）
        viol_loss = max(0.0, -margin_loss)
        viol_m1 = max(0.0, -margin_m1)
        viol_m2 = max(0.0, -margin_m2) if num_metrics > 1 else 0.0
        violation_cost = (
            float(barrier_weights["loss"]) * viol_loss
            + float(barrier_weights["metric1"]) * viol_m1
        )
        if num_metrics > 1:
            violation_cost += float(barrier_weights.get("metric2", 0.0)) * viol_m2
        per_trial_violation_costs.append(violation_cost)

        # 单次试验 margin utility（加权安全余量）
        util_loss = max(0.0, margin_loss)
        util_m1 = max(0.0, margin_m1)
        util_m2 = max(0.0, margin_m2) if num_metrics > 1 else 0.0
        margin_utility = (
            float(perf_weights["loss"]) * util_loss
            + float(perf_weights["metric1"]) * util_m1
        )
        if num_metrics > 1:
            margin_utility += float(perf_weights.get("metric2", 0.0)) * util_m2
        per_trial_margin_utilities.append(margin_utility)

        # 安全通过判定：所有指标 margin >= 0
        is_safe = (margin_loss >= 0.0) and (margin_m1 >= 0.0)
        if num_metrics > 1:
            is_safe = is_safe and (margin_m2 >= 0.0)
        per_trial_safe.append(is_safe)

    safe_rate = sum(per_trial_safe) / max(n, 1)
    unsafe_count = n - sum(per_trial_safe)

    # tail violation: violation cost 从大到小排序，取最坏 k 个平均
    sorted_violations = sorted(per_trial_violation_costs, reverse=True)
    tail_violation_cvar = float(np.mean(sorted_violations[:tail_k]))

    # tail margin: margin utility 从小到大排序，取最坏 k 个平均
    sorted_margins = sorted(per_trial_margin_utilities)
    tail_margin_score = float(np.mean(sorted_margins[:tail_k]))

    # 均值性能分数
    mean_perf_score = float(np.mean(per_trial_margin_utilities))

    # 尾部各指标均值（最坏 k 个试验）
    violation_indices = sorted(range(n), key=lambda i: per_trial_violation_costs[i], reverse=True)[:tail_k]
    tail_loss_mean = float(np.mean([trials[i]["loss"] for i in violation_indices]))
    tail_m1_mean = float(np.mean([trials[i]["metric1"] for i in violation_indices]))
    tail_m2_mean = (
        float(np.mean([trials[i].get("metric2", trials[i]["metric1"]) for i in violation_indices]))
        if num_metrics > 1 else tail_m1_mean
    )

    return {
        "n": n,
        "tail_k": tail_k,
        "safe_rate": safe_rate,
        "unsafe_count": unsafe_count,
        "tail_violation_cvar": tail_violation_cvar,
        "tail_margin_score": tail_margin_score,
        "mean_perf_score": mean_perf_score,
        "tail_loss_mean": tail_loss_mean,
        "tail_acc_mean": tail_m1_mean,
        "tail_f1_mean": tail_m2_mean,
        "per_trial_violation_costs": per_trial_violation_costs,
        "per_trial_margin_utilities": per_trial_margin_utilities,
        "per_trial_safe": per_trial_safe,
        "per_trial_margins_loss": per_trial_margins_loss,
        "per_trial_margins_m1": per_trial_margins_m1,
        "per_trial_margins_m2": per_trial_margins_m2,
    }


def _compute_wilson_lower_bound(successes, total, z=1.96):
    """计算 Wilson 置信区间的下界（用于 safe_rate 的置信修正）。"""
    if total == 0:
        return 0.0
    p_hat = successes / total
    denom = 1.0 + z * z / total
    center = p_hat + z * z / (2.0 * total)
    spread = z * math.sqrt((p_hat * (1.0 - p_hat) + z * z / (4.0 * total)) / total)
    return (center - spread) / denom


def _bootstrap_cvar_ucb(violation_costs, tail_k, n_bootstrap=NOISE_STAGE_BOOTSTRAP_ITERATIONS, confidence=0.95):
    """对 violation_costs 做 bootstrap 重采样，估计 CVaR 的均值和上置信界。

    种子由输入数据哈希生成，保证相同输入可复现、不同输入独立采样。
    """
    arr = np.array(violation_costs, dtype=np.float64)
    seed = hash(arr.tobytes()) % (2**32)
    rng = np.random.default_rng(seed)
    n = len(arr)
    bootstrap_cvars = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        sorted_sample = np.sort(sample)[::-1]
        cvar = float(np.mean(sorted_sample[:tail_k]))
        bootstrap_cvars.append(cvar)
    bootstrap_cvars = np.array(bootstrap_cvars)
    cvar_mean = float(np.mean(bootstrap_cvars))
    cvar_ucb = float(np.percentile(bootstrap_cvars, confidence * 100))
    return cvar_mean, cvar_ucb


def _log_rounded_box(log_fn, lines, indent="  ", min_inner_width=8):
    """圆角框日志：宽度随最长一行自动伸缩（按字符数，等宽字体下中英文可能对不齐属正常现象）。"""
    stripped = [str(x) for x in lines]
    w = max((len(s) for s in stripped), default=0)
    w = max(w, min_inner_width)
    bar = "─" * (w + 4)
    log_fn(f"{indent}╭{bar}╮")
    for s in stripped:
        log_fn(f"{indent}│ {s.ljust(w)} │")
    log_fn(f"{indent}╰{bar}╯")


def _write_rounded_box_to_file(f, lines, indent=""):
    """与 _log_rounded_box 相同规则，写入文本文件。"""
    stripped = [str(x) for x in lines]
    w = max((len(s) for s in stripped), default=0)
    w = max(w, 8)
    bar = "─" * (w + 4)
    f.write(f"{indent}╭{bar}╮\n")
    for s in stripped:
        f.write(f"{indent}│ {s.ljust(w)} │\n")
    f.write(f"{indent}╰{bar}╯\n")


def _noise_log_major_rule(log_fn, title, width=68):
    """二阶段噪声日志：小节顶栏（写入 pruning_search_log）。"""
    bar = "═" * width
    log_fn("")
    log_fn(bar)
    log_fn(title if title.startswith(" ") else f"  {title}")
    log_fn(bar)


def _noise_block_title(log_fn, title):
    """块标题，便于扫读。"""
    log_fn("")
    log_fn(f"  【{title}】")


def _shortlist_status_zh(status):
    """候选列表状态 → 中文（保留英文键）。"""
    mapping = {
        "not-eligible": "未达入列条件（not-eligible）",
        "disabled": "已禁用（disabled）",
        "added": "新加入（added）",
        "updated": "已更新（updated）",
        "duplicate-kept": "重复项保留原条（duplicate-kept）",
        "trimmed": "截断剔出（trimmed）",
    }
    return mapping.get(status, status)


def _warmup_status_zh(status):
    m = {"constant": "恒定（constant）", "warmup": "预热（warmup）", "normal": "常规（normal）"}
    return m.get(status, status)


def _noise_stage_status_zh(status, ok_token, no_stable_token):
    m = {
        ok_token: "正常（ok）",
        no_stable_token: "无稳定可行解（no_stable_feasible_candidate）",
    }
    return m.get(status, status)


class NoiseRLModule:
    """Standalone second-stage noise RL module.

    Follows the same pattern as FinalEvaluationModule: receives the evaluator
    and encapsulates all stage-2 noise RL logic (training, evaluation, plotting).
    """

    def __init__(self, evaluator):
        self.evaluator = evaluator

    def run(self, fixed_gelu, fixed_softmax, fixed_label, fixed_source):
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
            NOISE_STAGE_CONT_DIM,
            NOISE_STAGE_PREV_ACTION_EMBED_DIM,
            NOISE_STAGE_ACTION_DIMS,
            NOISE_STAGE_TRAINING_CURVE_PATH,
            NOISE_STAGE_ENTROPY_CURVE_PATH,
            PPO_UPDATE_INTERVAL,
            PPO_VALUE_COEF,
            REWARD_THRESHOLD,
            REWARD_DENSE_SCALE,
            REWARD_COST_WEIGHT,
            REWARD_SAFETY_BONUS,
            REWARD_CLIP_MIN,
            REWARD_CLIP_MAX,
            REWARD_NORMALIZATION_SCALE,
            BUDGET_DEVIATION_SCALE,
            HISTORY_MASK_VALUE,
            DIFF_REWARD_SCALE_ACC,
            DIFF_REWARD_POWER,
            LOG_BARRIER_VIOLATION_SCALE,
            LOG_BARRIER_VIOLATION_STEEPNESS,
            LOG_BARRIER_SATISFACTION_SCALE,
            LSTM_POS_DIM,
            LSTM_PROJ_DIM,
            GTRXL_ENTROPY_LOWER_BOUND,
            GTRXL_MINI_BATCH_EPISODES,
            USE_VALIDATION_FOR_REWARD,
            VALUE_CLIP_RANGE,
            GTrXLBlock,
            STEP_INFO_CHUNK_SIZE,
            REWARD_DROP_WARNING_THRESHOLD,
        )

        ev = self.evaluator
        noise_progress_dir = getattr(
            ev,
            "noise_stage_progress_dir",
            NOISE_STAGE_PROGRESS_DIR,
        )
        noise_training_curve_path = getattr(
            ev,
            "noise_stage_training_curve_path",
            NOISE_STAGE_TRAINING_CURVE_PATH,
        )
        noise_entropy_curve_path = getattr(
            ev,
            "noise_stage_entropy_curve_path",
            NOISE_STAGE_ENTROPY_CURVE_PATH,
        )
        stage2_total_episodes = int(
            getattr(ev, "stage2_rl_episodes", NOISE_STAGE_PPO_MAX_EPISODES)
        )
        _num_metrics_for_log = ev.get_num_metrics()
        _metric_names_for_log = ev.get_metric_names()
        _noise_log_major_rule(
            ev.log,
            "阶段5 · 二阶段噪声强化学习（Phase 5 · Second-stage noise RL）",
        )
        ev.log(
            f"  ▸ 固定 GELU / Softmax 来源（source）: {fixed_source}    "
            f"标签（label）: {fixed_label}"
        )
        ev.log(f"  ▸ GELU 离散阶数向量:   {np.asarray(fixed_gelu, dtype=int).tolist()}")
        ev.log(f"  ▸ Softmax 离散阶数向量: {np.asarray(fixed_softmax, dtype=int).tolist()}")
        ev.log(
            f"  ▸ 指标维度 num_metrics={_num_metrics_for_log}    "
            f"指标名 metric names={_metric_names_for_log}"
        )

        fixed_gelu = np.asarray(fixed_gelu, dtype=int)
        fixed_softmax = np.asarray(fixed_softmax, dtype=int)
        exact_baseline_gelu, exact_baseline_softmax = ev.get_stage1_exact_baseline_configuration()
        exact_baseline_gelu = np.asarray(exact_baseline_gelu, dtype=int)
        exact_baseline_softmax = np.asarray(exact_baseline_softmax, dtype=int)
        cost_reference_noise_config = ev._get_max_noise_configuration()

        # 性能 baseline 使用低风险噪声配置（每类噪声取最大 scaling factor = 噪声最小）
        baseline_noise_config = _get_low_risk_noise_configuration(ev)
        # worst-case 使用最大噪声配置（每类噪声取最小 scaling factor = 噪声最大）
        worst_case_noise_config = _get_worst_case_noise_configuration(ev)

        reward_reference_split = ev.get_reward_reference_split_name()

        def _copy_repeat_summary(summary):
            return {
                key: (
                    [dict(item) for item in value]
                    if key == "trials"
                    else value
                )
                for key, value in summary.items()
            }

        def _log_repeat_baseline(label, stats):
            ev.log(
                f"{label} · 重复 repeats={stats['n']} · 子集 split={stats['split_name']}"
            )
            if _num_metrics_for_log > 1:
                ev.log(
                    "     "
                    f"{ev._fmt_metrics(stats['loss_mean'], stats['p_mean'], stats['s_mean'])}  "
                    f"│ 标准差 std: loss={stats['loss_std']:.4f}, m1={stats['p_std']:.4f}, m2={stats['s_std']:.4f}"
                )
            else:
                ev.log(
                    "     "
                    f"{ev._fmt_metrics(stats['loss_mean'], stats['p_mean'], stats['s_mean'])}  "
                    f"│ 标准差 std: loss={stats['loss_std']:.4f}, m1={stats['p_std']:.4f}"
                )

        # Stage-2 性能 baseline：使用固定 Stage-1 配置 + 低风险参考噪声
        baseline_reference_stats = ev.evaluate_model_with_attention_noise_repeated(
            fixed_gelu,
            fixed_softmax,
            **baseline_noise_config,
            repeats=NOISE_STAGE_BASELINE_REPEATS,
            split=reward_reference_split,
        )
        if ev.has_dataset_split("val_holdout"):
            baseline_holdout_stats = ev.evaluate_model_with_attention_noise_repeated(
                fixed_gelu,
                fixed_softmax,
                **baseline_noise_config,
                repeats=NOISE_STAGE_BASELINE_REPEATS,
                split="val_holdout",
            )
        else:
            baseline_holdout_stats = _copy_repeat_summary(baseline_reference_stats)
            baseline_holdout_stats["split_name"] = "val_holdout"

        # Worst-case 评估：使用固定 Stage-1 配置 + 最大噪声配置（所有 scaling factor 取最小值）
        worst_reference_stats = ev.evaluate_model_with_attention_noise_repeated(
            fixed_gelu,
            fixed_softmax,
            **worst_case_noise_config,
            repeats=NOISE_STAGE_BASELINE_REPEATS,
            split=reward_reference_split,
        )
        if ev.has_dataset_split("val_holdout"):
            worst_holdout_stats = ev.evaluate_model_with_attention_noise_repeated(
                fixed_gelu,
                fixed_softmax,
                **worst_case_noise_config,
                repeats=NOISE_STAGE_BASELINE_REPEATS,
                split="val_holdout",
            )
        else:
            worst_holdout_stats = _copy_repeat_summary(worst_reference_stats)
            worst_holdout_stats["split_name"] = "val_holdout"

        cost_reference_tot_c, cost_reference_breakdown = ev.get_noise_simulated_cost(**cost_reference_noise_config)

        _noise_block_title(ev.log, "性能基线 · 固定 Stage-1 + 低风险参考噪声（Noise-stage baseline）")
        ev.log("  ▸ 与上阶段对齐的 GELU / Softmax：")
        ev.log(f"     GELU    {fixed_gelu.tolist()}")
        ev.log(f"     Softmax {fixed_softmax.tolist()}")
        ev.log("  ▸ 低风险缩放因子配置（low-risk noise config, SF 取噪声最小侧）：")
        for _bk, _bv in baseline_noise_config.items():
            ev.log(f"     {_bk}: {np.asarray(_bv, dtype=int).tolist()}")
        ev.log(f"  ▸ 子集 {reward_reference_split}（奖励与搜索约束所用，reward / search limits）：")
        _log_repeat_baseline("  基线（Baseline）", baseline_reference_stats)
        if ev.has_dataset_split("val_holdout"):
            ev.log("  ▸ 子集 val_holdout（留出集约束 holdout limits）：")
            _log_repeat_baseline("  基线（Baseline）", baseline_holdout_stats)

        _noise_block_title(ev.log, "最坏情形 · 固定 Stage-1 + 最强噪声（Worst-case noise）")
        ev.log("  ▸ 说明：各类噪声 SF 取允许集合最小值 → 扰动最强。")
        ev.log("  ▸ 最坏缩放因子配置（worst-case noise config）：")
        for _bk, _bv in worst_case_noise_config.items():
            ev.log(f"     {_bk}: {np.asarray(_bv, dtype=int).tolist()}")
        ev.log(f"  ▸ 子集 {reward_reference_split}：")
        _log_repeat_baseline("  最坏（Worst）", worst_reference_stats)
        if ev.has_dataset_split("val_holdout"):
            ev.log("  ▸ 子集 val_holdout：")
            _log_repeat_baseline("  最坏（Worst）", worst_holdout_stats)

        _noise_block_title(ev.log, "成本参考（Cost reference, 仅归一化）")
        ev.log(
            f"  ▸ 聚合成本（noise cost）: {cost_reference_tot_c:.2f}"
            f"    │    分项 breakdown: {cost_reference_breakdown}"
        )

        base_loss = baseline_reference_stats["loss_mean"]
        base_p = baseline_reference_stats["p_mean"]
        base_s = baseline_reference_stats["s_mean"]
        worst_loss = worst_reference_stats["loss_mean"]
        worst_p = worst_reference_stats["p_mean"]
        worst_s = worst_reference_stats["s_mean"]

        # 动态计算约束：见 NOISE_STAGE_DYNAMIC_LIMIT_QUARTILE
        search_limits = _compute_dynamic_limits(base_loss, base_p, base_s, worst_loss, worst_p, worst_s)
        holdout_limits = (
            _compute_dynamic_limits(
                baseline_holdout_stats["loss_mean"],
                baseline_holdout_stats["p_mean"],
                baseline_holdout_stats["s_mean"],
                worst_holdout_stats["loss_mean"],
                worst_holdout_stats["p_mean"],
                worst_holdout_stats["s_mean"],
            )
            if ev.has_dataset_split("val_holdout")
            else dict(search_limits)
        )
        limit_loss = float(search_limits["loss"])
        limit_p = float(search_limits["metric1"])
        limit_s = float(search_limits["metric2"])
        _noise_block_title(
            ev.log,
            f"搜索侧动态约束（Search constraints, "
            f"quartile={NOISE_STAGE_DYNAMIC_LIMIT_QUARTILE} on {reward_reference_split}）",
        )
        ev.log(f"  ▸ 基线（baseline）: {ev._fmt_metrics(base_loss, base_p, base_s)}")
        ev.log(f"  ▸ 最坏（worst）:   {ev._fmt_metrics(worst_loss, worst_p, worst_s)}")
        ev.log(f"  ▸ 界限（limit）:   {ev._fmt_constraints(limit_loss, limit_p, limit_s)}")
        if ev.has_dataset_split("val_holdout"):
            _noise_block_title(
                ev.log,
                "留出集动态约束（Holdout constraints, "
                f"quartile={NOISE_STAGE_DYNAMIC_LIMIT_QUARTILE} on val_holdout）",
            )
            ev.log(
                f"  ▸ 基线（baseline）: "
                f"{ev._fmt_metrics(baseline_holdout_stats['loss_mean'], baseline_holdout_stats['p_mean'], baseline_holdout_stats['s_mean'])}"
            )
            ev.log(
                f"  ▸ 最坏（worst）:   "
                f"{ev._fmt_metrics(worst_holdout_stats['loss_mean'], worst_holdout_stats['p_mean'], worst_holdout_stats['s_mean'])}"
            )
            ev.log(
                f"  ▸ 界限（limit）:   "
                f"{ev._fmt_constraints(holdout_limits['loss'], holdout_limits['metric1'], holdout_limits['metric2'])}"
            )
        training_hparams = {
            "performance_baseline_source": "stage1_fixed_low_risk_noise",
            "cost_reference_source": "max_noise_cost_reference",
            "gtrxl_d_model": NOISE_STAGE_GTRXL_D_MODEL,
            "gtrxl_n_heads": NOISE_STAGE_GTRXL_N_HEADS,
            "gtrxl_n_layers": NOISE_STAGE_GTRXL_N_LAYERS,
            "gtrxl_d_ff": NOISE_STAGE_GTRXL_D_FF,
            "gtrxl_dropout": NOISE_STAGE_GTRXL_DROPOUT,
            "ppo_max_episodes": stage2_total_episodes,
            "ppo_update_interval": PPO_UPDATE_INTERVAL,
            "ppo_eps_clip": NOISE_STAGE_PPO_EPS_CLIP,
            "ppo_k_epochs": NOISE_STAGE_PPO_K_EPOCHS,
            "ppo_value_coef": PPO_VALUE_COEF,
            "gtrxl_warmup_mode": NOISE_STAGE_GTRXL_WARMUP_MODE,
            "gtrxl_warmup_updates": NOISE_STAGE_GTRXL_WARMUP_UPDATES,
            "gtrxl_short_warmup_updates": NOISE_STAGE_GTRXL_SHORT_WARMUP_UPDATES,
            "gtrxl_mini_batch_episodes": GTRXL_MINI_BATCH_EPISODES,
            # 训练期/评估期评估次数分离
            "train_dense_eval_repeats": NOISE_STAGE_TRAIN_DENSE_EVAL_REPEATS,
            "train_terminal_eval_repeats": NOISE_STAGE_TRAIN_TERMINAL_EVAL_REPEATS,
            "dense_reward_mode": NOISE_STAGE_DENSE_REWARD_MODE,
            "train_terminal_reward_mode": NOISE_STAGE_TRAIN_TERMINAL_REWARD_MODE,
            "dense_completion_policy": NOISE_STAGE_DENSE_COMPLETION_POLICY,
            "mc_samples_legacy": NOISE_STAGE_MC_SAMPLES,
            "dynamic_limit_quartile": float(NOISE_STAGE_DYNAMIC_LIMIT_QUARTILE),
            "baseline_repeats": NOISE_STAGE_BASELINE_REPEATS,
            "online_baseline_repeats": NOISE_STAGE_ONLINE_BASELINE_REPEATS,
            "best_test_repeats": NOISE_STAGE_BEST_TEST_REPEATS,
            "best_test_trigger_margin": NOISE_STAGE_BEST_TEST_TRIGGER_MARGIN,
            "best_test_safe_rate_min": NOISE_STAGE_BEST_TEST_SAFE_RATE_MIN,
            "best_test_cvar_max": NOISE_STAGE_BEST_TEST_CVAR_MAX,
            "best_test_max_entry_score_drop": NOISE_STAGE_BEST_TEST_MAX_ENTRY_SCORE_DROP,
            "best_test_entry_std_multiplier": NOISE_STAGE_BEST_TEST_ENTRY_STD_MULTIPLIER,
            "progress_plot_interval": NOISE_STAGE_PROGRESS_SAVE_INTERVAL,
            "progress_plot_dir": noise_progress_dir,
            "stability_thresholds": {
                split: dict(values)
                for split, values in NOISE_STAGE_STABILITY_THRESHOLDS.items()
            },
            "stability_proxy_std_ref": NOISE_STAGE_STABILITY_PROXY_STD_REF,
            "stability_penalty_scale": NOISE_STAGE_STABILITY_PENALTY_SCALE,
            "reward_diff_enabled": False,
            "keep_cost_reward_when_violating": True,
            "cancel_dense_reward_on_violation": False,
            "final_reward_alpha_perf": NOISE_STAGE_FINAL_REWARD_ALPHA_PERF,
            "final_reward_alpha_cost": NOISE_STAGE_FINAL_REWARD_ALPHA_COST,
            "perf_weight_loss": NOISE_STAGE_PERF_WEIGHT_LOSS,
            "perf_weight_metric1": NOISE_STAGE_PERF_WEIGHT_M1,
            "perf_weight_metric2": NOISE_STAGE_PERF_WEIGHT_M2,
            "barrier_weight_loss": NOISE_STAGE_BARRIER_WEIGHT_LOSS,
            "barrier_weight_metric1": NOISE_STAGE_BARRIER_WEIGHT_M1,
            "barrier_weight_metric2": NOISE_STAGE_BARRIER_WEIGHT_M2,
            "stability_weight": NOISE_STAGE_STABILITY_PENALTY_SCALE,
            "dense_reward_shaping_scale": NOISE_STAGE_DENSE_REWARD_SHAPING_SCALE,
            "budget_decay_fraction": NOISE_STAGE_BUDGET_DECAY_FRACTION,
            "mc_base_samples": NOISE_STAGE_MC_BASE_SAMPLES,
            "mc_extra_samples": NOISE_STAGE_MC_EXTRA_SAMPLES,
            "mc_margin_threshold": NOISE_STAGE_MC_MARGIN_THRESHOLD,
            "tail_alpha_eval": NOISE_STAGE_TAIL_ALPHA_EVAL,
            "tail_train_k_min": NOISE_STAGE_TAIL_TRAIN_K_MIN,
            "tail_margin_weight": NOISE_STAGE_TAIL_MARGIN_WEIGHT,
            "tail_violation_weight": NOISE_STAGE_TAIL_VIOLATION_WEIGHT,
            "safe_rate_gap_weight": NOISE_STAGE_SAFE_RATE_GAP_WEIGHT,
            "mean_perf_weight": NOISE_STAGE_MEAN_PERF_WEIGHT,
            "cost_weight": NOISE_STAGE_COST_WEIGHT,
            "train_safe_rate_target": NOISE_STAGE_TRAIN_SAFE_RATE_TARGET,
            "window_topk": NOISE_STAGE_WINDOW_TOPK,
            "mean_aux_loss_coef": NOISE_STAGE_MEAN_AUX_LOSS_COEF,
        }
        _noise_block_title(ev.log, "训练超参与评估设定摘要（Training hyperparameters）")
        ev.log(
            "  ▸ 策略网络 GTrXL（GTrXL trunk）: "
            f"d_model={NOISE_STAGE_GTRXL_D_MODEL}, heads={NOISE_STAGE_GTRXL_N_HEADS}, "
            f"layers={NOISE_STAGE_GTRXL_N_LAYERS}, d_ff={NOISE_STAGE_GTRXL_D_FF}, "
            f"dropout={NOISE_STAGE_GTRXL_DROPOUT}"
        )
        ev.log(
            "  ▸ PPO: "
            f"max_episodes={stage2_total_episodes}, eps_clip={NOISE_STAGE_PPO_EPS_CLIP}, "
            f"k_epochs={NOISE_STAGE_PPO_K_EPOCHS}, warmup_mode={NOISE_STAGE_GTRXL_WARMUP_MODE}, "
            f"warmup_updates={NOISE_STAGE_GTRXL_WARMUP_UPDATES}, "
            f"short_warmup_updates={NOISE_STAGE_GTRXL_SHORT_WARMUP_UPDATES}"
        )
        ev.log(
            "  ▸ 训练期评估（train-time eval）: "
            f"dense 重复 dense_repeats={NOISE_STAGE_TRAIN_DENSE_EVAL_REPEATS}, "
            f"终止重复 terminal_repeats={NOISE_STAGE_TRAIN_TERMINAL_EVAL_REPEATS}, "
            f"dense 模式 dense_reward_mode={NOISE_STAGE_DENSE_REWARD_MODE}, "
            f"终止模式 terminal_reward_mode={NOISE_STAGE_TRAIN_TERMINAL_REWARD_MODE}, "
            f"补全策略 completion_policy={NOISE_STAGE_DENSE_COMPLETION_POLICY}  │  "
            f"预算衰减 budget_decay fraction={NOISE_STAGE_BUDGET_DECAY_FRACTION}"
        )
        ev.log(
            "  ▸ 终局奖励权重（final reward blending）: "
            f"alpha_perf={NOISE_STAGE_FINAL_REWARD_ALPHA_PERF}, "
            f"alpha_cost={NOISE_STAGE_FINAL_REWARD_ALPHA_COST}"
        )
        ev.log(
            "  ▸ 性能权 perf_weights: "
            f"loss={NOISE_STAGE_PERF_WEIGHT_LOSS}, m1={NOISE_STAGE_PERF_WEIGHT_M1}, m2={NOISE_STAGE_PERF_WEIGHT_M2}"
            "  │  屏障权 barrier_weights: "
            f"loss={NOISE_STAGE_BARRIER_WEIGHT_LOSS}, m1={NOISE_STAGE_BARRIER_WEIGHT_M1}, m2={NOISE_STAGE_BARRIER_WEIGHT_M2}"
        )
        ev.log(
            "  ▸ 奖励塑形（reward shaping）: 差分 diff=关闭（off）, "
            "违约保留成本项 keep_cost_on_violation=True, "
            "违约取消稠密项 cancel_dense_on_violation=False, "
            f"稠密比例 dense_scale={NOISE_STAGE_DENSE_REWARD_SHAPING_SCALE}"
        )
        ev.log(
            "  ▸ 候选复试（confirm）repeats="
            f"{NOISE_STAGE_CONFIRM_REPEATS}  │  决赛复试（finalist）repeats="
            f"{NOISE_STAGE_FINALIST_REPEATS}, shortlist 容量={NOISE_STAGE_SHORTLIST_SIZE}  │  "
            f"进度图间隔 progress_every={NOISE_STAGE_PROGRESS_SAVE_INTERVAL}, 目录={noise_progress_dir}"
        )
        ev.log(
            "  ▸ 稳定性阈值（stability thresholds）: "
            f"search={NOISE_STAGE_STABILITY_THRESHOLDS['search']}, "
            f"holdout={NOISE_STAGE_STABILITY_THRESHOLDS['holdout']}, "
            f"proxy 参考 std_ref={NOISE_STAGE_STABILITY_PROXY_STD_REF}, "
            f"惩罚系数 penalty_scale={NOISE_STAGE_STABILITY_PENALTY_SCALE}"
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
            """根据回合号返回所属分片文件路径"""
            chunk_start = ((episode_1based - 1) // STEP_INFO_CHUNK_SIZE) * STEP_INFO_CHUNK_SIZE + 1
            chunk_end = chunk_start + STEP_INFO_CHUNK_SIZE - 1
            return os.path.join(
                noise_step_info_details_dir,
                f"noise_ppo_step_info_{chunk_start}-{chunk_end}.txt",
            )

        def _open_noise_chunk(episode_1based):
            """打开当前回合所属的分片文件（如需切换则关闭旧文件并打开新文件）"""
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
                f"=== 噪声 PPO 逐步诊断（Noise PPO step info）· 回合区间 episodes {chunk_start}-{chunk_end} ===\n\n"
            )
            noise_step_info_chunk_file[0] = f
            noise_step_info_chunk_idx[0] = new_idx
            return f

        original_total_episodes = getattr(ev, "total_episodes", stage2_total_episodes)
        ev.total_episodes = stage2_total_episodes
        ev._reset_runtime_ppo_state()
        # 辅助 critic head 的独立归一化器（与 return_normalizer 解耦，避免梯度尺度不匹配）
        from layer_importance_evaluator import RunningMeanStd as _RunningMeanStd
        mean_perf_normalizer = _RunningMeanStd()
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
            noise_stage_cont_dim=NOISE_STAGE_CONT_DIM,
            noise_stage_action_dims=NOISE_STAGE_ACTION_DIMS,
        ).to(ev.device)
        optimizer = optim.Adam(noise_net.parameters(), lr=ev.ppo_lr_initial)
        noise_ppo_update_count = 0

        class _NoiseRLEvaluatorWrapper:
            def __init__(wrapper_self, evaluator, fixed_gelu, fixed_softmax, split_name=None, use_train=None):
                wrapper_self.evaluator = evaluator
                wrapper_self.fixed_gelu = np.asarray(fixed_gelu, dtype=int)
                wrapper_self.fixed_softmax = np.asarray(fixed_softmax, dtype=int)
                if split_name is not None:
                    wrapper_self.split_name = split_name
                elif use_train is None:
                    wrapper_self.split_name = "train"
                else:
                    wrapper_self.split_name = "train" if use_train else "validation_full"

            def evaluate_noise_model(wrapper_self, **noise_kwargs):
                return wrapper_self.evaluator.evaluate_model_with_attention_noise(
                    wrapper_self.fixed_gelu,
                    wrapper_self.fixed_softmax,
                    split=wrapper_self.split_name,
                    **noise_kwargs,
                )

        rl_evaluator = _NoiseRLEvaluatorWrapper(
            ev,
            fixed_gelu=fixed_gelu,
            fixed_softmax=fixed_softmax,
            use_train=(not USE_VALIDATION_FOR_REWARD),
        )
        use_fixed_search_reward = (
            USE_VALIDATION_FOR_REWARD
            and getattr(ev, "dataset_key", None) == "mrpc"
            and ev.has_dataset_split("val_search_full")
        )
        if USE_VALIDATION_FOR_REWARD:
            if use_fixed_search_reward:
                online_reward_split = reward_reference_split
                proxy_baseline_stats = _copy_repeat_summary(baseline_reference_stats)
            else:
                ev.refresh_validation_proxy(window_index=0, stage_label="Stage-2 Noise RL")
                online_reward_split = ev.get_online_reward_split_name()
                proxy_baseline_stats = ev.evaluate_model_with_attention_noise_repeated(
                    fixed_gelu,
                    fixed_softmax,
                    **baseline_noise_config,
                    repeats=NOISE_STAGE_ONLINE_BASELINE_REPEATS,
                    split=online_reward_split,
                )
            rl_evaluator.split_name = online_reward_split
            ev.log(
                f"  [信息] 在线奖励子集（online reward split）: {online_reward_split}  "
                f"（性能基线: Stage-1 固定 + 低风险噪声；成本尺度: 最大噪声配置）"
            )
        else:
            online_reward_split = "train"
            proxy_baseline_stats = _copy_repeat_summary(baseline_reference_stats)

        # ---- 训练期 proxy-specific baseline / limits ----
        # limit 必须基于与训练评估相同的数据集计算，否则 val_search_full 上的约束
        # 无法正确作用于 val_proxy 子集（由于采样方差，二者统计量可能有系统性偏差）。
        proxy_base_loss = float(proxy_baseline_stats["loss_mean"])
        proxy_base_p = float(proxy_baseline_stats["p_mean"])
        proxy_base_s = float(proxy_baseline_stats["s_mean"])

        if online_reward_split != reward_reference_split:
            # 在 proxy 上评估 worst-case，用于计算 proxy 专属 dynamic limits
            proxy_worst_stats = ev.evaluate_model_with_attention_noise_repeated(
                fixed_gelu,
                fixed_softmax,
                **worst_case_noise_config,
                repeats=NOISE_STAGE_ONLINE_BASELINE_REPEATS,
                split=online_reward_split,
            )
            proxy_worst_loss = float(proxy_worst_stats["loss_mean"])
            proxy_worst_p = float(proxy_worst_stats["p_mean"])
            proxy_worst_s = float(proxy_worst_stats["s_mean"])
            proxy_limits = _compute_dynamic_limits(
                proxy_base_loss, proxy_base_p, proxy_base_s,
                proxy_worst_loss, proxy_worst_p, proxy_worst_s,
            )
            _noise_block_title(
                ev.log,
                f"训练用代理子集动态约束（Training proxy limits on {online_reward_split}）",
            )
            ev.log(f"  ▸ 代理基线（proxy baseline）: {ev._fmt_metrics(proxy_base_loss, proxy_base_p, proxy_base_s)}")
            ev.log(f"  ▸ 代理最坏（proxy worst）:   {ev._fmt_metrics(proxy_worst_loss, proxy_worst_p, proxy_worst_s)}")
            ev.log(
                f"  ▸ 代理界限（proxy limit）:   "
                f"{ev._fmt_constraints(proxy_limits['loss'], proxy_limits['metric1'], proxy_limits['metric2'])}"
            )
        else:
            # online_reward_split == reward_reference_split，直接复用 search_limits
            proxy_limits = dict(search_limits)

        env = _NoiseOptEnv(
            ev.total_layers,
            cost_reference_tot_c,
            (proxy_base_loss, proxy_base_p, proxy_base_s),
            rl_evaluator,
            fixed_gelu=fixed_gelu,
            fixed_softmax=fixed_softmax,
            constraint_limits=proxy_limits,
            num_metrics=ev.get_num_metrics(),
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
            reward_threshold=REWARD_THRESHOLD,
            reward_dense_scale=REWARD_DENSE_SCALE,
            reward_cost_weight=REWARD_COST_WEIGHT,
            reward_safety_bonus=REWARD_SAFETY_BONUS,
            reward_clip_min=REWARD_CLIP_MIN,
            reward_clip_max=REWARD_CLIP_MAX,
            reward_normalization_scale=REWARD_NORMALIZATION_SCALE,
            budget_deviation_scale=BUDGET_DEVIATION_SCALE,
            diff_reward_scale_acc=DIFF_REWARD_SCALE_ACC,
            diff_reward_power=DIFF_REWARD_POWER,
            log_barrier_violation_scale=LOG_BARRIER_VIOLATION_SCALE,
            log_barrier_violation_steepness=LOG_BARRIER_VIOLATION_STEEPNESS,
            log_barrier_satisfaction_scale=LOG_BARRIER_SATISFACTION_SCALE,
            final_reward_alpha_perf=NOISE_STAGE_FINAL_REWARD_ALPHA_PERF,
            final_reward_alpha_cost=NOISE_STAGE_FINAL_REWARD_ALPHA_COST,
            perf_weight_loss=NOISE_STAGE_PERF_WEIGHT_LOSS,
            perf_weight_m1=NOISE_STAGE_PERF_WEIGHT_M1,
            perf_weight_m2=NOISE_STAGE_PERF_WEIGHT_M2,
            barrier_weight_loss=NOISE_STAGE_BARRIER_WEIGHT_LOSS,
            barrier_weight_m1=NOISE_STAGE_BARRIER_WEIGHT_M1,
            barrier_weight_m2=NOISE_STAGE_BARRIER_WEIGHT_M2,
            mc_samples=NOISE_STAGE_MC_SAMPLES,
            stability_weight=NOISE_STAGE_STABILITY_PENALTY_SCALE,
            stability_proxy_std_ref=NOISE_STAGE_STABILITY_PROXY_STD_REF,
            budget_decay_fraction=NOISE_STAGE_BUDGET_DECAY_FRACTION,
            mc_extra_samples=NOISE_STAGE_MC_EXTRA_SAMPLES,
            mc_margin_threshold=NOISE_STAGE_MC_MARGIN_THRESHOLD,
            dense_reward_shaping_scale=NOISE_STAGE_DENSE_REWARD_SHAPING_SCALE,
            train_dense_eval_repeats=NOISE_STAGE_TRAIN_DENSE_EVAL_REPEATS,
            train_terminal_eval_repeats=NOISE_STAGE_TRAIN_TERMINAL_EVAL_REPEATS,
            dense_reward_mode=NOISE_STAGE_DENSE_REWARD_MODE,
            train_terminal_reward_mode=NOISE_STAGE_TRAIN_TERMINAL_REWARD_MODE,
            dense_completion_policy=NOISE_STAGE_DENSE_COMPLETION_POLICY,
        )
        env.prev_episode_metrics = {
            "loss": float(proxy_baseline_stats["loss_mean"]),
            "metric1": float(proxy_baseline_stats["p_mean"]),
            "metric2": float(proxy_baseline_stats["s_mean"]),
            "cost": float(cost_reference_tot_c),
        }
        ev.log(
            f"  [信息] 奖励打分（reward scoring）: "
            f"成本区间 cost_bounds=({env._cost_lower_bound:.2f}, {env._cost_upper_bound:.2f}), "
            f"perf 权重 alpha_perf={env._final_reward_alpha_perf:.2f}, "
            f"cost 权重 alpha_cost={env._final_reward_alpha_cost:.2f}, "
            f"稠密塑形 dense_shaping_scale={env._dense_reward_shaping_scale:.2f}"
        )
        buffer = _NoiseRecurrentRolloutBuffer()

        episode_returns = []
        episode_raw_final_rewards = []
        episode_final_selection_scores = []
        episode_dense_reward_totals = []
        episode_losses = []
        episode_metric1s = []
        episode_metric2s = []
        episode_entropies = []
        stability_proxies = []
        stability_penalties = []
        # 尾部风险指标追踪
        episode_safe_rates = []
        episode_tail_violation_cvars = []
        episode_tail_margin_scores = []
        # 候选确认曲线追踪
        confirm_window_indices = []
        confirm_search_safe_rates = []
        confirm_holdout_safe_rates = []
        confirm_search_tail_cvars = []
        confirm_holdout_tail_cvars = []
        best_final_selection_score = float("-inf")
        best_cost = float("inf")
        best_noise_config = None
        incumbent_best_noise_config = None
        incumbent_best_signature = None
        incumbent_mean_score = float("-inf")
        incumbent_history = []
        qualification_cache = {}
        qualification_trigger_count = 0
        qualification_attempt_count = 0
        qualification_reuse_count = 0
        qualification_pass_count = 0
        qualification_fail_count = 0
        window_best_score = float("-inf")
        window_best_cost = float("inf")
        window_best_noise_config = None
        window_top_candidates = []
        noise_scaling_keys = tuple(
            key for key in cost_reference_noise_config.keys() if key.endswith("scaling_factors")
        )
        num_metrics = ev.get_num_metrics()
        ev_perf_weights = _resolve_stage2_metric_weights(
            num_metrics,
            {"loss": NOISE_STAGE_PERF_WEIGHT_LOSS, "metric1": NOISE_STAGE_PERF_WEIGHT_M1, "metric2": NOISE_STAGE_PERF_WEIGHT_M2},
            "perf_weights",
        )
        ev_barrier_weights = _resolve_stage2_metric_weights(
            num_metrics,
            {"loss": NOISE_STAGE_BARRIER_WEIGHT_LOSS, "metric1": NOISE_STAGE_BARRIER_WEIGHT_M1, "metric2": NOISE_STAGE_BARRIER_WEIGHT_M2},
            "barrier_weights",
        )
        stability_thresholds = {
            split: dict(values)
            for split, values in NOISE_STAGE_STABILITY_THRESHOLDS.items()
        }
        search_constraint_baseline_stats = baseline_reference_stats
        split_baseline_stats = {
            "search": _copy_repeat_summary(search_constraint_baseline_stats),
            "holdout": _copy_repeat_summary(baseline_holdout_stats),
        }
        qualification_search_split = (
            "val_search_full" if ev.has_dataset_split("val_search_full") else reward_reference_split
        )
        search_best_noise_config = None
        joint_best_noise_config = None
        stable_search_best_noise_config = None
        stable_joint_best_noise_config = None
        shortlist_candidates = []
        shortlist_update_count = 0

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

        def _candidate_signature(candidate):
            return tuple(
                tuple(np.asarray(candidate[key], dtype=int).tolist())
                for key in noise_scaling_keys
            )

        def _get_split_baseline_tuple(metric_prefix):
            baseline_stats = split_baseline_stats[metric_prefix]
            return (
                float(baseline_stats["loss_mean"]),
                float(baseline_stats["p_mean"]),
                float(baseline_stats["s_mean"]),
            )

        def _get_split_metric_sum(candidate, metric_prefix):
            metric_sum = float(candidate.get(f"{metric_prefix}_metric1", -float("inf")))
            if num_metrics > 1:
                metric_sum += float(candidate.get(f"{metric_prefix}_metric2", -float("inf")))
            return metric_sum

        # 排序键：显式字典序 tail -> mean -> cost（文档 7.3）
        def _split_sort_key(candidate, metric_prefix):
            return (
                not candidate.get(f"{metric_prefix}_risk_feasible", False),
                float(candidate.get(f"{metric_prefix}_tail_violation_cvar", float("inf"))),
                -float(candidate.get(f"{metric_prefix}_tail_margin_score", -float("inf"))),
                -float(candidate.get(f"{metric_prefix}_mean_perf_score", -float("inf"))),
                float(candidate.get("cost", float("inf"))),
                float(candidate.get(f"{metric_prefix}_stability_score", float("inf"))),
            )

        def _stable_split_sort_key(candidate, metric_prefix):
            return (
                not candidate.get(f"{metric_prefix}_risk_feasible", False),
                not candidate.get(f"{metric_prefix}_stability_ok", False),
                float(candidate.get(f"{metric_prefix}_tail_violation_cvar", float("inf"))),
                -float(candidate.get(f"{metric_prefix}_tail_margin_score", -float("inf"))),
                -float(candidate.get(f"{metric_prefix}_mean_perf_score", -float("inf"))),
                float(candidate.get("cost", float("inf"))),
                float(candidate.get(f"{metric_prefix}_stability_score", float("inf"))),
            )

        def _joint_metric_sum(candidate):
            metric_sum = (
                float(candidate.get("search_metric1", -float("inf")))
                + float(candidate.get("holdout_metric1", -float("inf")))
            )
            if num_metrics > 1:
                metric_sum += (
                    float(candidate.get("search_metric2", -float("inf")))
                    + float(candidate.get("holdout_metric2", -float("inf")))
                )
            return 0.5 * metric_sum

        def _joint_sort_key(candidate):
            # 联合排序键：取两个 split 的 worst case
            search_feasible = candidate.get("search_risk_feasible", False)
            holdout_feasible = candidate.get("holdout_risk_feasible", False)
            joint_feasible = search_feasible and holdout_feasible
            worst_cvar = max(
                float(candidate.get("search_tail_violation_cvar", float("inf"))),
                float(candidate.get("holdout_tail_violation_cvar", float("inf"))),
            )
            worst_margin = min(
                float(candidate.get("search_tail_margin_score", -float("inf"))),
                float(candidate.get("holdout_tail_margin_score", -float("inf"))),
            )
            worst_perf = min(
                float(candidate.get("search_mean_perf_score", -float("inf"))),
                float(candidate.get("holdout_mean_perf_score", -float("inf"))),
            )
            return (
                not joint_feasible,
                worst_cvar,
                -worst_margin,
                -worst_perf,
                float(candidate.get("cost", float("inf"))),
                float(candidate.get("stability_score", float("inf"))),
            )

        def _is_better_split_candidate(candidate, incumbent, metric_prefix):
            if candidate is None:
                return False
            if incumbent is None:
                return True
            return _split_sort_key(candidate, metric_prefix) < _split_sort_key(
                incumbent, metric_prefix
            )

        def _is_better_stable_split_candidate(candidate, incumbent, metric_prefix):
            if candidate is None:
                return False
            if incumbent is None:
                return True
            return _stable_split_sort_key(candidate, metric_prefix) < _stable_split_sort_key(
                incumbent, metric_prefix
            )

        def _is_better_joint_candidate(candidate, incumbent):
            if candidate is None:
                return False
            if incumbent is None:
                return True
            return _joint_sort_key(candidate) < _joint_sort_key(incumbent)

        def _annotate_split_stability(stats, metric_prefix, threshold_cfg, baseline_stats):
            repeats = max(1, int(stats[f"{metric_prefix}_repeats"]))
            sqrt_repeats = float(np.sqrt(repeats))
            components = [
                ("loss", "loss_std", "loss_sem_ratio", "loss_mean"),
                ("metric1", "metric1_std", "metric1_sem_ratio", "p_mean"),
            ]
            if num_metrics > 1:
                components.append(("metric2", "metric2_std", "metric2_sem_ratio", "s_mean"))

            ratios = []
            precision_ratios = []
            stability_ok = True
            precision_ok = True
            for metric_name, std_key, sem_ratio_key, baseline_key in components:
                stats_key = f"{metric_prefix}_{std_key}"
                threshold = float(threshold_cfg[std_key])
                std_value = float(stats[stats_key])
                ratio = std_value / max(threshold, 1e-8)
                stats[f"{metric_prefix}_{metric_name}_stability_ratio"] = float(ratio)
                stats[f"{metric_prefix}_{metric_name}_stability_threshold"] = threshold
                metric_ok = std_value <= threshold
                stats[f"{metric_prefix}_{metric_name}_stability_ok"] = bool(metric_ok)
                ratios.append(ratio)
                sem_value = std_value / max(sqrt_repeats, 1.0)
                stats[f"{metric_prefix}_{metric_name}_sem"] = float(sem_value)
                sem_threshold = threshold_cfg.get(sem_ratio_key)
                metric_precision_ok = True
                if sem_threshold is not None:
                    baseline_mean = abs(float(baseline_stats[baseline_key]))
                    sem_ratio = sem_value / max(baseline_mean, 1e-8)
                    precision_ratio = sem_ratio / max(float(sem_threshold), 1e-8)
                    stats[f"{metric_prefix}_{metric_name}_sem_ratio"] = float(sem_ratio)
                    stats[f"{metric_prefix}_{metric_name}_sem_threshold"] = float(sem_threshold)
                    stats[f"{metric_prefix}_{metric_name}_precision_ratio"] = float(precision_ratio)
                    metric_precision_ok = sem_ratio <= float(sem_threshold)
                    stats[f"{metric_prefix}_{metric_name}_precision_ok"] = bool(metric_precision_ok)
                    precision_ratios.append(precision_ratio)
                stability_ok = stability_ok and metric_ok and metric_precision_ok
                precision_ok = precision_ok and metric_precision_ok

            stats[f"{metric_prefix}_stability_score"] = float(np.mean(ratios)) if ratios else 0.0
            stats[f"{metric_prefix}_estimate_precision_score"] = (
                float(np.mean(precision_ratios)) if precision_ratios else 0.0
            )
            stats[f"{metric_prefix}_estimate_precision_ok"] = bool(precision_ok)
            stats[f"{metric_prefix}_stability_ok"] = bool(stability_ok)
            return stats

        def _evaluate_candidate_split(
            split_name,
            noise_kwargs,
            metric_prefix,
            repeats,
            threshold_cfg,
            constraint_lims=None,
            cost_value=None,
            risk_safe_rate_min=NOISE_STAGE_BEST_TEST_SAFE_RATE_MIN,
            risk_cvar_max=NOISE_STAGE_BEST_TEST_CVAR_MAX,
            score_baseline_metrics=None,
            compute_single_shot_scores=False,
            is_finalist=False,
        ):
            summary = ev.evaluate_model_with_attention_noise_repeated(
                fixed_gelu,
                fixed_softmax,
                repeats=repeats,
                split=split_name,
                **noise_kwargs,
            )
            stats = {
                f"{metric_prefix}_loss": float(summary["loss_mean"]),
                f"{metric_prefix}_loss_std": float(summary["loss_std"]),
                f"{metric_prefix}_loss_min": float(summary["loss_min"]),
                f"{metric_prefix}_loss_max": float(summary["loss_max"]),
                f"{metric_prefix}_loss_range": float(summary["loss_range"]),
                f"{metric_prefix}_metric1": float(summary["p_mean"]),
                f"{metric_prefix}_metric1_std": float(summary["p_std"]),
                f"{metric_prefix}_metric1_min": float(summary["p_min"]),
                f"{metric_prefix}_metric1_max": float(summary["p_max"]),
                f"{metric_prefix}_metric1_range": float(summary["p_range"]),
                f"{metric_prefix}_metric2": float(summary["s_mean"]),
                f"{metric_prefix}_metric2_std": float(summary["s_std"]),
                f"{metric_prefix}_metric2_min": float(summary["s_min"]),
                f"{metric_prefix}_metric2_max": float(summary["s_max"]),
                f"{metric_prefix}_metric2_range": float(summary["s_range"]),
                f"{metric_prefix}_repeats": int(summary["n"]),
            }
            # 尾部风险指标
            if constraint_lims is not None:
                raw_trials = summary.get("trials")
                if raw_trials is not None:
                    # 重映射键名：评估器返回 "p"/"s"，tail 指标计算需要 "metric1"/"metric2"
                    trials = [
                        {"loss": t["loss"], "metric1": t["p"], "metric2": t["s"]}
                        for t in raw_trials
                    ]
                else:
                    trials = [
                        {"loss": summary["loss_mean"], "metric1": summary["p_mean"], "metric2": summary["s_mean"]}
                    ] * int(summary["n"])
                confirm_k_min = NOISE_STAGE_TAIL_CONFIRM_K_MIN
                tail_info = _compute_tail_metrics_from_trials(
                    trials, constraint_lims,
                    _get_split_baseline_tuple(metric_prefix),
                    ev_perf_weights, ev_barrier_weights, num_metrics,
                    k_min=confirm_k_min,
                )
                stats[f"{metric_prefix}_safe_rate"] = tail_info["safe_rate"]
                stats[f"{metric_prefix}_tail_k"] = tail_info["tail_k"]
                stats[f"{metric_prefix}_tail_violation_cvar"] = tail_info["tail_violation_cvar"]
                stats[f"{metric_prefix}_tail_margin_score"] = tail_info["tail_margin_score"]
                stats[f"{metric_prefix}_mean_perf_score"] = tail_info["mean_perf_score"]
                stats[f"{metric_prefix}_tail_loss_mean"] = tail_info["tail_loss_mean"]
                stats[f"{metric_prefix}_tail_acc_mean"] = tail_info["tail_acc_mean"]
                stats[f"{metric_prefix}_tail_f1_mean"] = tail_info["tail_f1_mean"]
                stats[f"{metric_prefix}_unsafe_count"] = tail_info["unsafe_count"]
                # risk_feasible 判定
                if is_finalist:
                    sr_min = NOISE_STAGE_FINALIST_SAFE_RATE_MIN
                    cvar_max = NOISE_STAGE_FINALIST_CVAR_MAX
                else:
                    sr_min = NOISE_STAGE_CONFIRM_SAFE_RATE_MIN
                    cvar_max = NOISE_STAGE_CONFIRM_CVAR_MAX
                stats[f"{metric_prefix}_risk_feasible"] = bool(
                    tail_info["safe_rate"] >= sr_min
                    and tail_info["tail_violation_cvar"] <= cvar_max
                )
                # finalist 置信修正
                if is_finalist and repeats >= 16:
                    n_safe = int(round(tail_info["safe_rate"] * tail_info["n"]))
                    sr_lb95 = _compute_wilson_lower_bound(n_safe, tail_info["n"])
                    stats[f"{metric_prefix}_safe_rate_lb95"] = sr_lb95
                    cvar_mean, cvar_ucb95 = _bootstrap_cvar_ucb(
                        tail_info["per_trial_violation_costs"],
                        tail_info["tail_k"],
                    )
                    stats[f"{metric_prefix}_tail_violation_cvar_ucb95"] = cvar_ucb95
                    stats[f"{metric_prefix}_tail_violation_cvar_mean"] = cvar_mean
                    stats[f"{metric_prefix}_risk_feasible"] = bool(
                        tail_info["safe_rate"] >= sr_min
                        and sr_lb95 >= NOISE_STAGE_FINALIST_SAFE_RATE_LB95_MIN
                        and tail_info["tail_violation_cvar"] <= cvar_max
                    )
            stats = _annotate_split_stability(
                stats,
                metric_prefix,
                threshold_cfg,
                split_baseline_stats[metric_prefix],
            )
            return stats

        def _finalize_candidate_annotations(candidate):
            candidate["joint_loss_mean"] = 0.5 * (
                float(candidate["search_loss"]) + float(candidate["holdout_loss"])
            )
            candidate["joint_metric_sum"] = float(_joint_metric_sum(candidate))
            candidate["stability_score"] = 0.5 * (
                float(candidate.get("search_stability_score", float("inf")))
                + float(candidate.get("holdout_stability_score", float("inf")))
            )
            candidate["stable_search_feasible"] = bool(
                candidate["search_ok"] and candidate.get("search_stability_ok", False)
            )
            candidate["stable_holdout_feasible"] = bool(
                candidate["holdout_ok"] and candidate.get("holdout_stability_ok", False)
            )
            candidate["stable_joint_feasible"] = bool(
                candidate["stable_search_feasible"] and candidate["stable_holdout_feasible"]
            )
            # 尾部风险可行性
            candidate["search_risk_feasible"] = candidate.get("search_risk_feasible", False)
            candidate["holdout_risk_feasible"] = candidate.get("holdout_risk_feasible", False)
            candidate["joint_risk_feasible"] = bool(
                candidate["search_risk_feasible"] and candidate["holdout_risk_feasible"]
            )
            return candidate

        def _upsert_shortlist_candidate(candidate):
            nonlocal shortlist_update_count
            shortlist_update_count += 1
            signature = _candidate_signature(candidate)
            action = "added"
            existing_index = None
            for idx, existing in enumerate(shortlist_candidates):
                if _candidate_signature(existing) == signature:
                    existing_index = idx
                    break

            if existing_index is not None:
                if _is_better_joint_candidate(candidate, shortlist_candidates[existing_index]):
                    shortlist_candidates[existing_index] = _clone_candidate(candidate)
                    action = "updated"
                else:
                    action = "duplicate-kept"
            else:
                shortlist_candidates.append(_clone_candidate(candidate))

            shortlist_candidates.sort(key=_joint_sort_key)
            if len(shortlist_candidates) > NOISE_STAGE_SHORTLIST_SIZE:
                del shortlist_candidates[NOISE_STAGE_SHORTLIST_SIZE:]

            retained = any(
                _candidate_signature(existing) == signature
                for existing in shortlist_candidates
            )
            if not retained and action in {"added", "updated"}:
                action = "trimmed"
            return action

        # 动态风险阈值（Dynamic risk thresholds）
        # 初始值为静态默认值，基线评估后根据其实际表现自动放宽，
        # 确保基线方案自身一定能通过检测，同时挑战者也以基线为参照。
        dynamic_best_test_risk_thresholds = {
            "search": {
                "safe_rate_min": NOISE_STAGE_BEST_TEST_SAFE_RATE_MIN,
                "cvar_max": NOISE_STAGE_BEST_TEST_CVAR_MAX,
            },
            "holdout": {
                "safe_rate_min": NOISE_STAGE_BEST_TEST_SAFE_RATE_MIN,
                "cvar_max": NOISE_STAGE_BEST_TEST_CVAR_MAX,
            },
        }

        def _evaluate_best_test_split(
            split_name,
            noise_kwargs,
            metric_prefix,
            repeats,
            threshold_cfg,
            constraint_lims,
            cost_value,
        ):
            summary = ev.evaluate_model_with_attention_noise_repeated(
                fixed_gelu,
                fixed_softmax,
                repeats=repeats,
                split=split_name,
                **noise_kwargs,
            )
            stats = {
                f"{metric_prefix}_loss": float(summary["loss_mean"]),
                f"{metric_prefix}_loss_std": float(summary["loss_std"]),
                f"{metric_prefix}_loss_min": float(summary["loss_min"]),
                f"{metric_prefix}_loss_max": float(summary["loss_max"]),
                f"{metric_prefix}_loss_range": float(summary["loss_range"]),
                f"{metric_prefix}_metric1": float(summary["p_mean"]),
                f"{metric_prefix}_metric1_std": float(summary["p_std"]),
                f"{metric_prefix}_metric1_min": float(summary["p_min"]),
                f"{metric_prefix}_metric1_max": float(summary["p_max"]),
                f"{metric_prefix}_metric1_range": float(summary["p_range"]),
                f"{metric_prefix}_metric2": float(summary["s_mean"]),
                f"{metric_prefix}_metric2_std": float(summary["s_std"]),
                f"{metric_prefix}_metric2_min": float(summary["s_min"]),
                f"{metric_prefix}_metric2_max": float(summary["s_max"]),
                f"{metric_prefix}_metric2_range": float(summary["s_range"]),
                f"{metric_prefix}_repeats": int(summary["n"]),
            }

            raw_trials = summary.get("trials")
            if raw_trials is not None:
                trials = [
                    {"loss": t["loss"], "metric1": t["p"], "metric2": t["s"]}
                    for t in raw_trials
                ]
            else:
                trials = [
                    {
                        "loss": summary["loss_mean"],
                        "metric1": summary["p_mean"],
                        "metric2": summary["s_mean"],
                    }
                ] * int(summary["n"])

            tail_info = _compute_tail_metrics_from_trials(
                trials,
                constraint_lims,
                _get_split_baseline_tuple(metric_prefix),
                ev_perf_weights,
                ev_barrier_weights,
                num_metrics,
                k_min=NOISE_STAGE_TAIL_CONFIRM_K_MIN,
            )
            stats[f"{metric_prefix}_safe_rate"] = tail_info["safe_rate"]
            stats[f"{metric_prefix}_tail_k"] = tail_info["tail_k"]
            stats[f"{metric_prefix}_tail_violation_cvar"] = tail_info["tail_violation_cvar"]
            stats[f"{metric_prefix}_tail_margin_score"] = tail_info["tail_margin_score"]
            stats[f"{metric_prefix}_mean_perf_score"] = tail_info["mean_perf_score"]
            stats[f"{metric_prefix}_tail_loss_mean"] = tail_info["tail_loss_mean"]
            stats[f"{metric_prefix}_tail_acc_mean"] = tail_info["tail_acc_mean"]
            stats[f"{metric_prefix}_tail_f1_mean"] = tail_info["tail_f1_mean"]
            stats[f"{metric_prefix}_unsafe_count"] = tail_info["unsafe_count"]
            _dyn_sr_min = dynamic_best_test_risk_thresholds[metric_prefix]["safe_rate_min"]
            _dyn_cvar_max = dynamic_best_test_risk_thresholds[metric_prefix]["cvar_max"]
            stats[f"{metric_prefix}_risk_feasible"] = bool(
                tail_info["safe_rate"] >= _dyn_sr_min
                and tail_info["tail_violation_cvar"] <= _dyn_cvar_max
            )

            reward_scores = []
            for trial in trials:
                reward_score, _reward_components = _compute_stage2_single_shot_reward_components(
                    loss=trial["loss"],
                    metric1=trial["metric1"],
                    metric2=trial["metric2"],
                    baseline_metrics=_get_split_baseline_tuple(metric_prefix),
                    constraint_limits=constraint_lims,
                    cost_value=float(cost_value),
                    cost_lower_bound=env._cost_lower_bound,
                    cost_upper_bound=env._cost_upper_bound,
                    perf_weights=ev_perf_weights,
                    barrier_weights=ev_barrier_weights,
                    num_metrics=num_metrics,
                    safe_rate_target=NOISE_STAGE_TRAIN_SAFE_RATE_TARGET,
                )
                reward_scores.append(float(reward_score))
            reward_scores_arr = np.asarray(reward_scores, dtype=np.float32)
            stats[f"{metric_prefix}_score_mean"] = float(np.mean(reward_scores_arr))
            stats[f"{metric_prefix}_score_std"] = float(np.std(reward_scores_arr))
            stats[f"{metric_prefix}_score_min"] = float(np.min(reward_scores_arr))
            stats[f"{metric_prefix}_score_max"] = float(np.max(reward_scores_arr))
            stats[f"{metric_prefix}_score_range"] = float(
                stats[f"{metric_prefix}_score_max"] - stats[f"{metric_prefix}_score_min"]
            )
            return _annotate_split_stability(
                stats,
                metric_prefix,
                threshold_cfg,
                split_baseline_stats[metric_prefix],
            )

        def _run_best_test_base(candidate_config, repeats, qualification_label):
            noise_kwargs = {
                key: value.copy()
                for key, value in candidate_config.items()
                if key.endswith("scaling_factors")
            }
            cost_value = float(candidate_config["cost"])
            search_stats = _evaluate_best_test_split(
                qualification_search_split,
                noise_kwargs,
                metric_prefix="search",
                repeats=repeats,
                threshold_cfg=stability_thresholds["search"],
                constraint_lims=search_limits,
                cost_value=cost_value,
            )
            search_ok = ev._candidate_meets_constraints(
                search_stats["search_loss"],
                search_stats["search_metric1"],
                search_stats["search_metric2"],
                search_limits["loss"],
                search_limits["metric1"],
                search_limits["metric2"],
            )

            evaluated_candidate = {
                key: (value.copy() if isinstance(value, np.ndarray) else value)
                for key, value in candidate_config.items()
            }
            evaluated_candidate.update({
                "reward": float(search_stats["search_score_mean"]),
                "raw_final_reward": float(search_stats["search_score_mean"]),
                "final_selection_score": float(search_stats["search_score_mean"]),
                "perf_score": float(search_stats["search_mean_perf_score"]),
                "cost_score": float(np.clip(
                    (float(env._cost_upper_bound) - cost_value)
                    / max(float(env._cost_upper_bound) - float(env._cost_lower_bound), 1e-8),
                    0.0,
                    1.0,
                )),
                "barrier_penalty": float(search_stats["search_tail_violation_cvar"]),
                "reward_components": {
                    "search_score_mean": float(search_stats["search_score_mean"]),
                    "search_score_std": float(search_stats["search_score_std"]),
                },
                "confirmation_label": qualification_label,
                "confirmed_repeats": int(repeats),
                "search_ok": bool(search_ok),
                "confirmed_episode": None,
                "confirmed_window": None,
            })
            evaluated_candidate.update(search_stats)

            if ev.has_dataset_split("val_holdout"):
                holdout_stats = _evaluate_best_test_split(
                    "val_holdout",
                    noise_kwargs,
                    metric_prefix="holdout",
                    repeats=repeats,
                    threshold_cfg=stability_thresholds["holdout"],
                    constraint_lims=holdout_limits,
                    cost_value=cost_value,
                )
                holdout_ok = ev._candidate_meets_constraints(
                    holdout_stats["holdout_loss"],
                    holdout_stats["holdout_metric1"],
                    holdout_stats["holdout_metric2"],
                    holdout_limits["loss"],
                    holdout_limits["metric1"],
                    holdout_limits["metric2"],
                )
            else:
                holdout_stats = {
                    "holdout_loss": float(search_stats["search_loss"]),
                    "holdout_loss_std": float(search_stats["search_loss_std"]),
                    "holdout_loss_min": float(search_stats["search_loss_min"]),
                    "holdout_loss_max": float(search_stats["search_loss_max"]),
                    "holdout_loss_range": float(search_stats["search_loss_range"]),
                    "holdout_metric1": float(search_stats["search_metric1"]),
                    "holdout_metric1_std": float(search_stats["search_metric1_std"]),
                    "holdout_metric1_min": float(search_stats["search_metric1_min"]),
                    "holdout_metric1_max": float(search_stats["search_metric1_max"]),
                    "holdout_metric1_range": float(search_stats["search_metric1_range"]),
                    "holdout_metric2": float(search_stats["search_metric2"]),
                    "holdout_metric2_std": float(search_stats["search_metric2_std"]),
                    "holdout_metric2_min": float(search_stats["search_metric2_min"]),
                    "holdout_metric2_max": float(search_stats["search_metric2_max"]),
                    "holdout_metric2_range": float(search_stats["search_metric2_range"]),
                    "holdout_repeats": int(search_stats["search_repeats"]),
                    "holdout_safe_rate": float(search_stats["search_safe_rate"]),
                    "holdout_tail_k": int(search_stats["search_tail_k"]),
                    "holdout_tail_violation_cvar": float(search_stats["search_tail_violation_cvar"]),
                    "holdout_tail_margin_score": float(search_stats["search_tail_margin_score"]),
                    "holdout_mean_perf_score": float(search_stats["search_mean_perf_score"]),
                    "holdout_tail_loss_mean": float(search_stats["search_tail_loss_mean"]),
                    "holdout_tail_acc_mean": float(search_stats["search_tail_acc_mean"]),
                    "holdout_tail_f1_mean": float(search_stats["search_tail_f1_mean"]),
                    "holdout_unsafe_count": int(search_stats["search_unsafe_count"]),
                    "holdout_risk_feasible": bool(search_stats["search_risk_feasible"]),
                    "holdout_score_mean": float(search_stats["search_score_mean"]),
                    "holdout_score_std": float(search_stats["search_score_std"]),
                    "holdout_score_min": float(search_stats["search_score_min"]),
                    "holdout_score_max": float(search_stats["search_score_max"]),
                    "holdout_score_range": float(search_stats["search_score_range"]),
                }
                holdout_stats = _annotate_split_stability(
                    holdout_stats,
                    "holdout",
                    stability_thresholds["holdout"],
                    split_baseline_stats["holdout"],
                )
                holdout_ok = bool(search_ok)

            evaluated_candidate.update(holdout_stats)
            evaluated_candidate["holdout_ok"] = bool(holdout_ok)
            evaluated_candidate["joint_ok"] = bool(search_ok and holdout_ok)
            evaluated_candidate = _finalize_candidate_annotations(evaluated_candidate)
            evaluated_candidate["base_qualification_passed"] = bool(
                evaluated_candidate["stable_joint_feasible"]
                and evaluated_candidate["joint_risk_feasible"]
            )
            return evaluated_candidate

        def _materialize_best_test_result(
            base_candidate,
            episode_idx,
            entry_score,
            qualification_label,
            from_cache=False,
        ):
            candidate = _clone_candidate(base_candidate)
            search_score_mean = float(candidate.get("search_score_mean", float("-inf")))
            search_score_std = float(candidate.get("search_score_std", 0.0))
            entry_score_value = (
                None if entry_score is None else float(entry_score)
            )
            entry_gap = (
                0.0 if entry_score_value is None else float(entry_score_value - search_score_mean)
            )
            entry_tol = max(
                float(NOISE_STAGE_BEST_TEST_MAX_ENTRY_SCORE_DROP),
                float(NOISE_STAGE_BEST_TEST_ENTRY_STD_MULTIPLIER) * search_score_std,
            )
            entry_score_ok = (
                True if entry_score_value is None else entry_gap <= entry_tol
            )
            candidate.update({
                "qualification_label": qualification_label,
                "qualification_from_cache": bool(from_cache),
                "qualification_entry_score": entry_score_value,
                "qualification_entry_gap": float(entry_gap),
                "qualification_entry_gap_tol": float(entry_tol),
                "qualification_entry_score_ok": bool(entry_score_ok),
                "qualification_passed": bool(
                    candidate.get("base_qualification_passed", False) and entry_score_ok
                ),
                "confirmation_label": qualification_label,
                "confirmed_episode": (
                    None if episode_idx is None or episode_idx < 0 else int(episode_idx) + 1
                ),
                "confirmed_window": None,
                "shortlist_status": "disabled",
            })
            return candidate

        def _log_best_test_result(candidate):
            cache_zh = (
                "命中缓存（cache-hit）"
                if candidate.get("qualification_from_cache")
                else "全新评估（fresh-eval）"
            )
            label_zh = candidate.get("qualification_label", "挑战者（challenger）")
            lines = [
                f"最优复试（BestTest） · {cache_zh} · {label_zh}",
                f"── 搜索子集（Search）: {qualification_search_split} ──",
                f"  {ev._fmt_metrics(candidate['search_loss'], candidate['search_metric1'], candidate['search_metric2'])}",
                (
                    f"  得分均值 score_mean={candidate.get('search_score_mean', 0.0):.4f}  ;  "
                    f"标准差 score_std={candidate.get('search_score_std', 0.0):.4f}  ;  "
                    f"安全率 safe_rate={candidate.get('search_safe_rate', 0.0):.3f}  ;  "
                    f"尾部 CVaR={candidate.get('search_tail_violation_cvar', 0.0):.4f}  ;  "
                    f"稳定项通过 stable={candidate.get('search_stability_ok', False)}"
                ),
                "── 留出集（Holdout）──",
                f"  {ev._fmt_metrics(candidate['holdout_loss'], candidate['holdout_metric1'], candidate['holdout_metric2'])}",
                (
                    f"  得分均值 score_mean={candidate.get('holdout_score_mean', 0.0):.4f}  ;  "
                    f"标准差 score_std={candidate.get('holdout_score_std', 0.0):.4f}  ;  "
                    f"安全率 safe_rate={candidate.get('holdout_safe_rate', 0.0):.3f}  ;  "
                    f"尾部 CVaR={candidate.get('holdout_tail_violation_cvar', 0.0):.4f}  ;  "
                    f"稳定项通过 stable={candidate.get('holdout_stability_ok', False)}"
                ),
            ]
            if candidate.get("qualification_entry_score") is not None:
                lines.append(
                    f"── 训练入围分（entry score）──  "
                    f"entry_score={candidate['qualification_entry_score']:.4f}, "
                    f"gap={candidate['qualification_entry_gap']:.4f}, "
                    f"tol={candidate['qualification_entry_gap_tol']:.4f}, "
                    f"entry_ok={candidate['qualification_entry_score_ok']}"
                )
            lines.append(
                f"── 裁定（verdict）──  "
                f"基础合格 base_passed={candidate.get('base_qualification_passed', False)}  ;  "
                f"总合格 qualification_passed={candidate.get('qualification_passed', False)}"
            )
            _log_rounded_box(ev.log, lines, indent="")

        def _qualify_challenger_candidate(
            candidate_config,
            episode_idx,
            entry_score,
            qualification_label,
        ):
            nonlocal qualification_trigger_count
            nonlocal qualification_attempt_count, qualification_reuse_count
            nonlocal qualification_pass_count, qualification_fail_count

            qualification_trigger_count += 1
            signature = _candidate_signature(candidate_config)
            cached_candidate = qualification_cache.get(signature)
            from_cache = cached_candidate is not None
            if from_cache:
                qualification_reuse_count += 1
                base_candidate = _clone_candidate(cached_candidate)
            else:
                qualification_attempt_count += 1
                base_candidate = _run_best_test_base(
                    candidate_config,
                    repeats=NOISE_STAGE_BEST_TEST_REPEATS,
                    qualification_label=qualification_label,
                )
                qualification_cache[signature] = _clone_candidate(base_candidate)
                if base_candidate.get("base_qualification_passed", False):
                    qualification_pass_count += 1
                else:
                    qualification_fail_count += 1

            candidate = _materialize_best_test_result(
                base_candidate,
                episode_idx=episode_idx,
                entry_score=entry_score,
                qualification_label=qualification_label,
                from_cache=from_cache,
            )
            confirm_window_indices.append(
                0 if episode_idx is None or episode_idx < 0 else int(episode_idx) + 1
            )
            confirm_search_safe_rates.append(float(candidate.get("search_safe_rate", 0.0)))
            confirm_holdout_safe_rates.append(float(candidate.get("holdout_safe_rate", 0.0)))
            confirm_search_tail_cvars.append(float(candidate.get("search_tail_violation_cvar", 0.0)))
            confirm_holdout_tail_cvars.append(float(candidate.get("holdout_tail_violation_cvar", 0.0)))
            _log_best_test_result(candidate)
            return candidate

        def confirm_noise_candidate(
            candidate_config,
            episode_idx,
            window_idx,
            repeats,
            confirmation_label,
            update_shortlist=False,
            is_finalist=False,
        ):
            nonlocal search_best_noise_config, joint_best_noise_config
            nonlocal stable_search_best_noise_config, stable_joint_best_noise_config

            if candidate_config is None or not ev.has_dataset_split("val_search_full"):
                return None

            noise_kwargs = {
                key: value.copy()
                for key, value in candidate_config.items()
                if key.endswith("scaling_factors")
            }
            search_stats = _evaluate_candidate_split(
                "val_search_full",
                noise_kwargs,
                metric_prefix="search",
                repeats=repeats,
                threshold_cfg=stability_thresholds["search"],
                constraint_lims=search_limits,
                is_finalist=is_finalist,
            )
            search_ok = ev._candidate_meets_constraints(
                search_stats["search_loss"],
                search_stats["search_metric1"],
                search_stats["search_metric2"],
                search_limits["loss"],
                search_limits["metric1"],
                search_limits["metric2"],
            )
            search_mc_eval = {
                "loss_std": float(search_stats["search_loss_std"]),
                "metric1_std": float(search_stats["search_metric1_std"]),
                "metric2_std": float(search_stats["search_metric2_std"]),
            }
            confirmed_raw_final_reward, confirmed_reward_components = env.score_reward_components(
                search_stats["search_loss"],
                search_stats["search_metric1"],
                search_stats["search_metric2"],
                float(candidate_config["cost"]),
                mc_eval=search_mc_eval,
                constraint_limits=search_limits,
            )

            confirmed_candidate = {
                key: (value.copy() if isinstance(value, np.ndarray) else value)
                for key, value in candidate_config.items()
            }
            confirmed_candidate.update({
                "reward": float(confirmed_raw_final_reward),
                "raw_final_reward": float(confirmed_raw_final_reward),
                "final_selection_score": float(confirmed_reward_components["final_selection_score"]),
                "perf_score": float(confirmed_reward_components["perf_score"]),
                "cost_score": float(confirmed_reward_components["cost_score"]),
                "barrier_penalty": float(confirmed_reward_components["barrier_penalty"]),
                "stability_proxy": float(confirmed_reward_components.get("stability_proxy", 0.0)),
                "stability_penalty": float(confirmed_reward_components.get("stability_penalty", 0.0)),
                "reward_components": dict(confirmed_reward_components),
                "confirmation_label": confirmation_label,
                "confirmed_repeats": int(max(1, int(repeats))),
                "search_ok": bool(search_ok),
                "confirmed_episode": int(episode_idx) + 1,
                "confirmed_window": int(window_idx) + 1,
            })
            confirmed_candidate.update(search_stats)

            if ev.has_dataset_split("val_holdout"):
                holdout_stats = _evaluate_candidate_split(
                    "val_holdout",
                    noise_kwargs,
                    metric_prefix="holdout",
                    repeats=repeats,
                    threshold_cfg=stability_thresholds["holdout"],
                    constraint_lims=holdout_limits,
                    is_finalist=is_finalist,
                )
                holdout_ok = ev._candidate_meets_constraints(
                    holdout_stats["holdout_loss"],
                    holdout_stats["holdout_metric1"],
                    holdout_stats["holdout_metric2"],
                    holdout_limits["loss"],
                    holdout_limits["metric1"],
                    holdout_limits["metric2"],
                )
            else:
                holdout_stats = {
                    "holdout_loss": float(search_stats["search_loss"]),
                    "holdout_loss_std": float(search_stats["search_loss_std"]),
                    "holdout_loss_min": float(search_stats["search_loss_min"]),
                    "holdout_loss_max": float(search_stats["search_loss_max"]),
                    "holdout_loss_range": float(search_stats["search_loss_range"]),
                    "holdout_metric1": float(search_stats["search_metric1"]),
                    "holdout_metric1_std": float(search_stats["search_metric1_std"]),
                    "holdout_metric1_min": float(search_stats["search_metric1_min"]),
                    "holdout_metric1_max": float(search_stats["search_metric1_max"]),
                    "holdout_metric1_range": float(search_stats["search_metric1_range"]),
                    "holdout_metric2": float(search_stats["search_metric2"]),
                    "holdout_metric2_std": float(search_stats["search_metric2_std"]),
                    "holdout_metric2_min": float(search_stats["search_metric2_min"]),
                    "holdout_metric2_max": float(search_stats["search_metric2_max"]),
                    "holdout_metric2_range": float(search_stats["search_metric2_range"]),
                    "holdout_repeats": int(search_stats["search_repeats"]),
                }
                holdout_stats = _annotate_split_stability(
                    holdout_stats,
                    "holdout",
                    stability_thresholds["holdout"],
                    split_baseline_stats["holdout"],
                )
                holdout_ok = bool(search_ok)

            confirmed_candidate.update(holdout_stats)
            confirmed_candidate["holdout_ok"] = bool(holdout_ok)
            confirmed_candidate["joint_ok"] = bool(search_ok and holdout_ok)
            confirmed_candidate = _finalize_candidate_annotations(confirmed_candidate)

            _confirm_lines = [
                f"噪声候选确认（confirm） · {confirmation_label}",
                "── 搜索集（Search）──",
                f"  指标 metrics: {ev._fmt_metrics(confirmed_candidate['search_loss'], confirmed_candidate['search_metric1'], confirmed_candidate['search_metric2'])}",
                (
                    f"  风险 risk: 安全率 safe_rate={confirmed_candidate.get('search_safe_rate', 0):.3f}, "
                    f"CVaR={confirmed_candidate.get('search_tail_violation_cvar', 0):.4f}, "
                    f"尾部余量 tail_margin={confirmed_candidate.get('search_tail_margin_score', 0):.4f}, "
                    f"风险可行 risk_feasible={confirmed_candidate.get('search_risk_feasible', False)}"
                ),
                (
                    f"  性能 perf: 均值分项 mean_perf={confirmed_candidate.get('search_mean_perf_score', 0):.4f}, "
                    f"成本 cost={confirmed_candidate['cost']:.2f}"
                ),
            ]
            sr_lb = confirmed_candidate.get("search_safe_rate_lb95")
            cvar_ucb = confirmed_candidate.get("search_tail_violation_cvar_ucb95")
            if sr_lb is not None:
                _confirm_lines.append(
                    f"  置信下界: Wilson LB={sr_lb:.4f}, CVaR UCB={cvar_ucb:.4f}"
                )
            if num_metrics > 1:
                _confirm_lines.append(
                    f"  统计 statistics: std(loss={confirmed_candidate['search_loss_std']:.4f}, "
                    f"m1={confirmed_candidate['search_metric1_std']:.4f}, "
                    f"m2={confirmed_candidate['search_metric2_std']:.4f}), "
                    f"稳定 stability_ok={confirmed_candidate['search_stability_ok']}, "
                    f"估计精度 precision_ok={confirmed_candidate['search_estimate_precision_ok']}"
                )
            else:
                _confirm_lines.append(
                    f"  统计 statistics: std(loss={confirmed_candidate['search_loss_std']:.4f}, "
                    f"m1={confirmed_candidate['search_metric1_std']:.4f}), "
                    f"稳定 stability_ok={confirmed_candidate['search_stability_ok']}, "
                    f"估计精度 precision_ok={confirmed_candidate['search_estimate_precision_ok']}"
                )
            if ev.has_dataset_split("val_holdout"):
                _confirm_lines.append("── 留出集（Holdout）──")
                _confirm_lines.append(
                    f"  指标 metrics: {ev._fmt_metrics(confirmed_candidate['holdout_loss'], confirmed_candidate['holdout_metric1'], confirmed_candidate['holdout_metric2'])}"
                )
                _confirm_lines.append(
                    f"  风险 risk: 安全率 safe_rate={confirmed_candidate.get('holdout_safe_rate', 0):.3f}, "
                    f"CVaR={confirmed_candidate.get('holdout_tail_violation_cvar', 0):.4f}, "
                    f"尾部余量 tail_margin={confirmed_candidate.get('holdout_tail_margin_score', 0):.4f}, "
                    f"风险可行 risk_feasible={confirmed_candidate.get('holdout_risk_feasible', False)}"
                )
                h_lb = confirmed_candidate.get("holdout_safe_rate_lb95")
                h_ucb = confirmed_candidate.get("holdout_tail_violation_cvar_ucb95")
                if h_lb is not None:
                    _confirm_lines.append(
                        f"  置信下界: Wilson LB={h_lb:.4f}, CVaR UCB={h_ucb:.4f}"
                    )
            _log_rounded_box(ev.log, _confirm_lines)

            if search_ok and _is_better_split_candidate(
                confirmed_candidate,
                search_best_noise_config,
                metric_prefix="search",
            ):
                search_best_noise_config = _clone_candidate(confirmed_candidate)
                ev.log(
                    f"  ★ 搜索侧最优（Noise search-best）更新 · 回合 {episode_idx + 1}: "
                    f"成本 cost={search_best_noise_config['cost']:.2f}, "
                    f"终选分 final_selection_score={search_best_noise_config['final_selection_score']:.4f}"
                )

            if confirmed_candidate["stable_search_feasible"] and _is_better_stable_split_candidate(
                confirmed_candidate,
                stable_search_best_noise_config,
                metric_prefix="search",
            ):
                stable_search_best_noise_config = _clone_candidate(confirmed_candidate)
                ev.log(
                    f"  ★ 稳定搜索最优（Noise stable search-best）· 回合 {episode_idx + 1}: "
                    f"成本 cost={stable_search_best_noise_config['cost']:.2f}, "
                    f"终选分 final_selection_score={stable_search_best_noise_config['final_selection_score']:.4f}, "
                    f"稳定分 stability_score={stable_search_best_noise_config['search_stability_score']:.4f}"
                )

            if confirmed_candidate["joint_ok"] and _is_better_joint_candidate(
                confirmed_candidate,
                joint_best_noise_config,
            ):
                joint_best_noise_config = _clone_candidate(confirmed_candidate)
                ev.log(
                    f"  ★ 联合最优（Noise joint-best）· 回合 {episode_idx + 1}: "
                    f"成本 cost={joint_best_noise_config['cost']:.2f}, "
                    f"终选分 final_selection_score={joint_best_noise_config['final_selection_score']:.4f}"
                )

            if confirmed_candidate["stable_joint_feasible"] and _is_better_joint_candidate(
                confirmed_candidate,
                stable_joint_best_noise_config,
            ):
                stable_joint_best_noise_config = _clone_candidate(confirmed_candidate)
                ev.log(
                    f"  ★ 稳定联合最优（Noise stable joint-best）· 回合 {episode_idx + 1}: "
                    f"成本 cost={stable_joint_best_noise_config['cost']:.2f}, "
                    f"终选分 final_selection_score={stable_joint_best_noise_config['final_selection_score']:.4f}, "
                    f"稳定分 stability_score={stable_joint_best_noise_config['stability_score']:.4f}"
                )

            shortlist_status = "not-eligible"
            if update_shortlist and confirmed_candidate["stable_joint_feasible"]:
                shortlist_status = _upsert_shortlist_candidate(confirmed_candidate)
            confirmed_candidate["shortlist_status"] = shortlist_status
            ev.log("  ▶ 稳定性裁定（stability verdict）:")
            ev.log(
                f"      搜索硬约束 search_ok={confirmed_candidate['search_ok']}, "
                f"留出硬约束 holdout_ok={confirmed_candidate['holdout_ok']}, "
                f"搜索稳定 stable_search={confirmed_candidate['stable_search_feasible']}, "
                f"留出稳定 stable_holdout={confirmed_candidate['stable_holdout_feasible']}, "
                f"联合稳定可行 stable_joint={confirmed_candidate['stable_joint_feasible']}, "
                f"短名单状态 shortlist={_shortlist_status_zh(shortlist_status)}"
            )
            return confirmed_candidate

        def _compute_training_stability_proxy(mc_eval):
            return _compute_stage2_stability_terms(
                mc_eval,
                NOISE_STAGE_STABILITY_PROXY_STD_REF,
                NOISE_STAGE_STABILITY_PENALTY_SCALE,
                num_metrics,
            )

        incumbent_cost_value, _incumbent_cost_breakdown = ev.get_noise_simulated_cost(
            **baseline_noise_config
        )
        initial_incumbent_candidate = {
            key: np.asarray(value, dtype=int).copy()
            for key, value in baseline_noise_config.items()
        }
        initial_incumbent_candidate["cost"] = float(incumbent_cost_value)
        initial_incumbent_base = _run_best_test_base(
            initial_incumbent_candidate,
            repeats=NOISE_STAGE_BEST_TEST_REPEATS,
            qualification_label="初始守擂基线（initial incumbent）",
        )
        qualification_cache[_candidate_signature(initial_incumbent_candidate)] = _clone_candidate(
            initial_incumbent_base
        )

        # ── 根据基线 + 最坏情形的实际评估结果动态放宽风险阈值 ──
        # safe_rate_min: 取 min(static, baseline) — 确保基线自身能通过
        # cvar_max: 取 max(static, baseline, worst) — 确保实际噪声配置有机会通过
        #   最坏情形是"所有噪声拉满"的极端参照，其 CVaR 代表可达的上限级别，
        #   挑战者只要优于最坏情形就应该有资格竞争。
        _bl_search_sr = float(initial_incumbent_base.get("search_safe_rate", 1.0))
        _bl_search_cvar = float(initial_incumbent_base.get("search_tail_violation_cvar", 0.0))
        _bl_holdout_sr = float(initial_incumbent_base.get("holdout_safe_rate", 1.0))
        _bl_holdout_cvar = float(initial_incumbent_base.get("holdout_tail_violation_cvar", 0.0))

        # 从最坏情形的重复评估数据中计算 tail CVaR（无需额外评估）
        def _compute_worst_case_cvar(worst_stats, constraint_lims, baseline_tuple):
            raw_trials = worst_stats.get("trials")
            if not raw_trials:
                return 0.0
            trials = [
                {"loss": t["loss"], "metric1": t["p"], "metric2": t["s"]}
                for t in raw_trials
            ]
            tail_info = _compute_tail_metrics_from_trials(
                trials, constraint_lims, baseline_tuple,
                ev_perf_weights, ev_barrier_weights, num_metrics,
                k_min=NOISE_STAGE_TAIL_CONFIRM_K_MIN,
            )
            return float(tail_info["tail_violation_cvar"])

        _worst_search_cvar = _compute_worst_case_cvar(
            worst_reference_stats, search_limits,
            _get_split_baseline_tuple("search"),
        )
        _worst_holdout_cvar = _compute_worst_case_cvar(
            worst_holdout_stats, holdout_limits,
            _get_split_baseline_tuple("holdout"),
        )

        dynamic_best_test_risk_thresholds["search"]["safe_rate_min"] = min(
            NOISE_STAGE_BEST_TEST_SAFE_RATE_MIN, _bl_search_sr,
        )
        dynamic_best_test_risk_thresholds["search"]["cvar_max"] = max(
            NOISE_STAGE_BEST_TEST_CVAR_MAX, _bl_search_cvar, _worst_search_cvar,
        )
        dynamic_best_test_risk_thresholds["holdout"]["safe_rate_min"] = min(
            NOISE_STAGE_BEST_TEST_SAFE_RATE_MIN, _bl_holdout_sr,
        )
        dynamic_best_test_risk_thresholds["holdout"]["cvar_max"] = max(
            NOISE_STAGE_BEST_TEST_CVAR_MAX, _bl_holdout_cvar, _worst_holdout_cvar,
        )
        ev.log(
            f"  [动态风险阈值 Dynamic Risk Thresholds] 基线+最坏情形评估后更新（update after baseline & worst eval）:\n"
            f"    search : safe_rate_min="
            f"{dynamic_best_test_risk_thresholds['search']['safe_rate_min']:.4f}"
            f" (静态 static={NOISE_STAGE_BEST_TEST_SAFE_RATE_MIN}, 基线 baseline={_bl_search_sr:.4f}), "
            f"cvar_max="
            f"{dynamic_best_test_risk_thresholds['search']['cvar_max']:.4f}"
            f" (静态 static={NOISE_STAGE_BEST_TEST_CVAR_MAX}, 基线 baseline={_bl_search_cvar:.4f}"
            f", 最坏 worst={_worst_search_cvar:.4f})\n"
            f"    holdout: safe_rate_min="
            f"{dynamic_best_test_risk_thresholds['holdout']['safe_rate_min']:.4f}"
            f" (静态 static={NOISE_STAGE_BEST_TEST_SAFE_RATE_MIN}, 基线 baseline={_bl_holdout_sr:.4f}), "
            f"cvar_max="
            f"{dynamic_best_test_risk_thresholds['holdout']['cvar_max']:.4f}"
            f" (静态 static={NOISE_STAGE_BEST_TEST_CVAR_MAX}, 基线 baseline={_bl_holdout_cvar:.4f}"
            f", 最坏 worst={_worst_holdout_cvar:.4f})"
        )

        # ── 基线方案无条件上台 ──
        # 基线是性能参照点，其合格标志强制置 True
        initial_incumbent_base["search_risk_feasible"] = True
        initial_incumbent_base["holdout_risk_feasible"] = True
        initial_incumbent_base["joint_risk_feasible"] = True
        initial_incumbent_base["base_qualification_passed"] = True
        # 同步更新资格缓存，避免与基线同签名的候选使用过期的旧标志
        qualification_cache[_candidate_signature(initial_incumbent_candidate)] = _clone_candidate(
            initial_incumbent_base
        )

        initial_incumbent_result = _materialize_best_test_result(
            initial_incumbent_base,
            episode_idx=None,
            entry_score=None,
            qualification_label="初始守擂基线（initial incumbent）",
            from_cache=False,
        )
        _log_best_test_result(initial_incumbent_result)

        # 基线无条件作为守擂（baseline unconditionally becomes incumbent）
        incumbent_best_noise_config = _clone_candidate(initial_incumbent_result)
        incumbent_best_signature = _candidate_signature(incumbent_best_noise_config)
        incumbent_mean_score = float(incumbent_best_noise_config["search_score_mean"])
        incumbent_history.append({
            "source": "initial_incumbent",
            "episode": 0,
            "mean_score": float(incumbent_mean_score),
            "cost": float(incumbent_best_noise_config["cost"]),
            "base_qualification_passed": True,
        })
        ev.log(
            f"  [最优复试 BestTest] 初始守擂已就绪（基线无条件上台 baseline unconditionally accepted）: "
            f"search 得分均值 mean_score={incumbent_mean_score:.4f}, "
            f"成本 cost={incumbent_best_noise_config['cost']:.2f}"
        )

        for episode in range(stage2_total_episodes):
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
            episode_mc_eval = None
            episode_stability_proxy = None
            episode_stability_penalty = 0.0
            buffer.start_episode()

            for step in range(ev.total_layers):
                layer_idx = env.current_layer
                cont_feat_np = env.get_continuous_features()
                cont_feat = torch.tensor(cont_feat_np, dtype=torch.float32, device=ev.device).view(1, 1, -1)
                layer_tensor = torch.tensor([[layer_idx]], dtype=torch.long, device=ev.device)

                seq_cont_feats.append(cont_feat)
                seq_layer_indices.append(layer_tensor)
                seq_prev_actions.append(prev_actions)

                cont_seq = torch.cat(seq_cont_feats, dim=1)
                layer_seq = torch.cat(seq_layer_indices, dim=1)
                prev_action_seq = torch.cat(seq_prev_actions, dim=1)

                actions, logprob, value, prob_list = noise_net.get_action_and_logprob(
                    cont_seq, layer_seq, prev_action_seq, return_probs=True
                )
                next_state, reward, done, info = env.step(*[a.item() for a in actions])
                reward_for_buffer = reward
                if done:
                    raw_final_reward = float(info.get("raw_final_reward", 0.0))
                    final_selection_score = float(
                        info.get("final_selection_score", raw_final_reward)
                    )

                    mc_eval_for_penalty = info.get("mc_eval") or {}
                    stability_proxy, stability_penalty = _compute_training_stability_proxy(
                        mc_eval_for_penalty
                    )
                    stability_proxies.append(stability_proxy)
                    stability_penalties.append(stability_penalty)

                    info["step_reward"] = reward
                    info["final_selection_score"] = final_selection_score
                    info["stability_proxy"] = stability_proxy
                    info["stability_penalty"] = stability_penalty

                    episode_raw_final_reward = raw_final_reward
                    episode_final_selection_score = final_selection_score
                    episode_mc_eval = mc_eval_for_penalty
                    episode_stability_proxy = stability_proxy
                    episode_stability_penalty = stability_penalty
                else:
                    info["step_reward"] = reward
                    info["final_selection_score"] = None
                    info["stability_proxy"] = None
                    info["stability_penalty"] = 0.0

                mc_eval = info.get("mc_eval") or {}
                step_info = {
                    "step_global": episode * ev.total_layers + step,
                    "episode_id": episode,
                    "layer_index": info["layer_index"],
                    "state_vector": state.tolist(),
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
                    "input_noise_config": info["input_noise_config"],
                    "wq_noise_config": info["wq_noise_config"],
                    "wk_noise_config": info["wk_noise_config"],
                    "wv_noise_config": info["wv_noise_config"],
                    "wo_noise_config": info["wo_noise_config"],
                    "wffn1_noise_config": info["wffn1_noise_config"],
                    "wffn2_noise_config": info["wffn2_noise_config"],
                    "current_lr": current_lr,
                    "current_entropy_coef": current_entropy,
                    "mc_samples": mc_eval.get("num_samples"),
                    "mc_loss_mean": mc_eval.get("loss_mean"),
                    "mc_loss_std": mc_eval.get("loss_std"),
                    "mc_metric1_mean": mc_eval.get("metric1_mean"),
                    "mc_metric1_std": mc_eval.get("metric1_std"),
                    "mc_metric2_mean": mc_eval.get("metric2_mean"),
                    "mc_metric2_std": mc_eval.get("metric2_std"),
                    "step_reward": info.get("step_reward"),
                    "dense_reward_step": info.get("dense_reward"),
                    "raw_final_reward": info.get("raw_final_reward"),
                    "final_selection_score": info.get("final_selection_score"),
                    "accumulated_dense_reward": info.get("accumulated_dense_reward"),
                    "stability_proxy": info.get("stability_proxy"),
                    "stability_penalty": info.get("stability_penalty"),
                    # dense probe 详情（文档修改方案 第8步）
                    "partial_probe_loss": info.get("partial_probe_loss"),
                    "partial_probe_metric1": info.get("partial_probe_metric1"),
                    "partial_probe_metric2": info.get("partial_probe_metric2"),
                    "partial_margin_loss": info.get("partial_margin_loss"),
                    "partial_margin_metric1": info.get("partial_margin_metric1"),
                    "partial_margin_metric2": info.get("partial_margin_metric2"),
                    "partial_completion_score": info.get("partial_completion_score"),
                    "partial_violation_penalty": info.get("partial_violation_penalty"),
                    "remaining_budget_ratio": info.get("remaining_budget_ratio"),
                    "completion_policy": info.get("completion_policy"),
                }
                step_infos.append(step_info)

                buffer.add_step(
                    cont_feat=torch.tensor(cont_feat_np, dtype=torch.float32),
                    layer_idx=layer_idx,
                    prev_actions=prev_actions.squeeze(0).squeeze(0).detach().cpu(),
                    actions=actions.detach().cpu(),
                    logprob=logprob.detach().cpu(),
                    reward=reward_for_buffer,
                    value=value.detach().cpu(),
                    done=float(done),
                    mean_perf_target=float(info.get("mean_perf_value", 0.0)),
                )

                prev_actions = actions.view(1, 1, -1).to(ev.device)
                episode_reward_raw += reward
                state = next_state

            buffer.end_episode()
            episode_returns.append(episode_reward_raw)
            episode_raw_final_rewards.append(
                episode_raw_final_reward if episode_raw_final_reward is not None else 0.0
            )
            episode_dense_reward_totals.append(float(env.accumulated_dense_reward))
            if env.current_episode_metrics is not None:
                episode_losses.append(env.current_episode_metrics["loss"])
                episode_metric1s.append(env.current_episode_metrics["metric1"])
                episode_metric2s.append(env.current_episode_metrics["metric2"])
            else:
                episode_losses.append(proxy_base_loss)
                episode_metric1s.append(proxy_base_p)
                episode_metric2s.append(proxy_base_s)

            # 追踪尾部风险指标
            _rc = env.last_reward_components or {}
            episode_safe_rates.append(float(_rc.get("safe_rate", 1.0)))
            episode_tail_violation_cvars.append(float(_rc.get("tail_violation_cvar", 0.0)))
            episode_tail_margin_scores.append(float(_rc.get("tail_margin_score", 0.0)))

            ev.update_reward_statistics(episode_reward_raw)
            chunk_f = _open_noise_chunk(episode + 1)
            chunk_f.write(
                f"── 回合 episode {episode + 1} ──  "
                f"回报 episode_return={episode_reward_raw:.4f}  ;  "
                f"原始终局 raw_final_reward="
                f"{(episode_raw_final_reward if episode_raw_final_reward is not None else 0.0):.4f}  ;  "
                f"稠密累计 dense_reward_total={env.accumulated_dense_reward:.4f}\n"
            )
            for si in step_infos:
                _write_noise_step_info(si, chunk_f)
                chunk_f.write("\n")
            chunk_f.flush()

            reward_components = (
                dict(env.last_reward_components)
                if getattr(env, "last_reward_components", None) is not None
                else None
            )
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
                    else (episode_raw_final_reward if episode_raw_final_reward is not None else episode_reward_raw)
                ),
                "episode_return": episode_reward_raw,
                "dense_reward_total": float(env.accumulated_dense_reward),
                "stability_proxy": episode_stability_proxy,
                "stability_penalty": episode_stability_penalty,
                "mc_eval": dict(episode_mc_eval) if episode_mc_eval is not None else None,
                "reward_components": reward_components,
            }
            if reward_components is not None:
                final_noise_config.update({
                    "perf_score": float(reward_components["perf_score"]),
                    "cost_score": float(reward_components["cost_score"]),
                    "barrier_penalty": float(reward_components["barrier_penalty"]),
                })

            episode_final_selection_score = float(final_noise_config["final_selection_score"])
            episode_final_selection_scores.append(episode_final_selection_score)
            if episode_final_selection_score > window_best_score or (
                episode_final_selection_score == window_best_score and env.accumulated_cost < window_best_cost
            ):
                window_best_score = episode_final_selection_score
                window_best_cost = env.accumulated_cost
                window_best_noise_config = {
                    key: (
                        value.copy()
                        if isinstance(value, np.ndarray)
                        else dict(value)
                        if isinstance(value, dict)
                        else value
                    )
                    for key, value in final_noise_config.items()
                }
            # 维护窗口 top-k 候选列表
            _wc = _clone_candidate(final_noise_config)
            window_top_candidates.append(_wc)
            window_top_candidates.sort(key=lambda c: -float(c.get("final_selection_score", -float("inf"))))
            if len(window_top_candidates) > NOISE_STAGE_WINDOW_TOPK:
                window_top_candidates = window_top_candidates[:NOISE_STAGE_WINDOW_TOPK]

            if episode_final_selection_score > best_final_selection_score or (
                episode_final_selection_score == best_final_selection_score and env.accumulated_cost < best_cost
            ):
                best_final_selection_score = episode_final_selection_score
                best_cost = env.accumulated_cost
                best_noise_config = {
                    key: (
                        value.copy()
                        if isinstance(value, np.ndarray)
                        else dict(value)
                        if isinstance(value, dict)
                        else value
                    )
                    for key, value in final_noise_config.items()
                }
                _noise_block_title(ev.log, f"回合 {episode + 1} · 训练过程新高（episode new best）")
                ev.log(
                    f"  ▸ 终选分 final_selection_score={episode_final_selection_score:.4f}  "
                    f"│  原始终局奖励 raw_final_reward="
                    f"{(episode_raw_final_reward if episode_raw_final_reward is not None else 0.0):.4f}  "
                    f"│  回合回报 episode_return={episode_reward_raw:.4f}  "
                    f"│  成本 cost={env.accumulated_cost:.2f}"
                )
                if episode_mc_eval is not None:
                    _mc_log = (
                        "  ▸ 蒙特卡洛诊断（MC eval）: "
                        f"损失 loss={episode_mc_eval['loss_mean']:.4f}±{episode_mc_eval['loss_std']:.4f}, "
                        f"指标1 m1={episode_mc_eval['metric1_mean']:.4f}±{episode_mc_eval['metric1_std']:.4f}"
                    )
                    if num_metrics > 1:
                        _mc_log += (
                            f", 指标2 m2={episode_mc_eval['metric2_mean']:.4f}"
                            f"±{episode_mc_eval['metric2_std']:.4f}"
                        )
                    ev.log(_mc_log)
                ev.log(f"     输入 input_noise:  {env.input_noise_config}")
                ev.log(f"     权重 wq:          {env.wq_noise_config}")
                ev.log(f"     权重 wk:          {env.wk_noise_config}")
                ev.log(f"     权重 wv:          {env.wv_noise_config}")
                ev.log(f"     权重 wo:          {env.wo_noise_config}")
                ev.log(f"     FFN1 wffn1:       {env.wffn1_noise_config}")
                ev.log(f"     FFN2 wffn2:       {env.wffn2_noise_config}")

            challenger_signature = _candidate_signature(final_noise_config)
            if (
                (incumbent_best_signature is None or challenger_signature != incumbent_best_signature)
                and episode_final_selection_score > (
                    incumbent_mean_score + float(NOISE_STAGE_BEST_TEST_TRIGGER_MARGIN)
                )
            ):
                challenger_candidate = _qualify_challenger_candidate(
                    final_noise_config,
                    episode_idx=episode,
                    entry_score=episode_final_selection_score,
                    qualification_label=(
                        f"第 {episode + 1} 回合挑战者（episode {episode + 1} challenger）"
                    ),
                )
                if challenger_candidate is not None and challenger_candidate.get("qualification_passed", False):
                    challenger_mean_score = float(challenger_candidate["search_score_mean"])
                    if incumbent_best_noise_config is None or challenger_mean_score > incumbent_mean_score:
                        had_incumbent = incumbent_best_noise_config is not None
                        prev_mean_score = (
                            float(challenger_mean_score)
                            if not had_incumbent
                            else float(incumbent_mean_score)
                        )
                        prev_cost = (
                            float(challenger_candidate["cost"])
                            if not had_incumbent
                            else float(incumbent_best_noise_config["cost"])
                        )
                        incumbent_best_noise_config = _clone_candidate(challenger_candidate)
                        incumbent_best_signature = challenger_signature
                        incumbent_mean_score = challenger_mean_score
                        incumbent_history.append({
                            "source": (
                                "challenger_initialization"
                                if not had_incumbent
                                else "challenger_update"
                            ),
                            "episode": int(episode) + 1,
                            "mean_score": float(challenger_mean_score),
                            "cost": float(challenger_candidate["cost"]),
                            "entry_score": float(episode_final_selection_score),
                        })
                        ev.log(
                            f"  ★ 守擂最优（incumbent best）已更新 · 回合 episode={episode + 1}: "
                            f"旧 search 均值 old_mean={prev_mean_score:.4f} → 新 new_mean={challenger_mean_score:.4f}, "
                            f"旧成本 old_cost={prev_cost:.2f} → 新 new_cost={challenger_candidate['cost']:.2f}"
                        )
                    else:
                        ev.log(
                            f"  [最优复试 BestTest] 挑战者虽总合格，但 search 均值 mean_score={challenger_mean_score:.4f} "
                            f"未超过守擂均值 incumbent_mean={incumbent_mean_score:.4f}，保留原 best。"
                        )

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
                    value_clip_range=VALUE_CLIP_RANGE,
                    mean_perf_normalizer=mean_perf_normalizer,
                )
                noise_ppo_update_count += 1
                buffer.clear()
                episode_entropies.append(entropy)
                avg_episode_return = np.mean(episode_returns[-PPO_UPDATE_INTERVAL:])
                avg_raw_final_reward = np.mean(
                    episode_raw_final_rewards[-PPO_UPDATE_INTERVAL:]
                )
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
                        f"PPO 窗口摘要 · 截至回合（through episode） {episode + 1}",
                        f"平均回报 mean episode return={avg_episode_return:.4f}  ;  "
                        f"平均原始终局奖励 mean raw final={avg_raw_final_reward:.4f}",
                        f"策略损失 policy_loss={policy_loss:.4f}  ;  "
                        f"价值损失 value_loss={value_loss:.4f}  ;  熵 entropy={entropy:.4f}",
                        (
                            f"GTrXL 调度（schedule）: LR={optimizer.param_groups[0]['lr']:.6f}, "
                            f"熵系数 entropy_coef={current_entropy:.6f}, "
                            f"更新序号 update#{noise_ppo_update_count}, "
                            f"模式 mode={NOISE_STAGE_GTRXL_WARMUP_MODE}, "
                            f"状态 state={_warmup_status_zh(warmup_status)}"
                        ),
                    ],
                )

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
                            "type": "噪声阶段奖励骤降",
                            "window": noise_ppo_update_count,
                            "prev_avg": float(noise_prev_avg_reward[0]),
                            "curr_avg": float(avg_raw_final_reward),
                            "drop": float(reward_drop),
                            "threshold": float(REWARD_DROP_WARNING_THRESHOLD),
                            "episode_range": (window_start_ep, window_end_ep),
                            "detail_files": involved_basenames,
                        }
                        noise_warnings.append(warn_msg)
                        # ── 立即输出完整警告（immediate detailed warning） ──
                        _warn_lines = [
                            f"⚠ 奖励骤降警告 · 第 {len(noise_warnings)} 条",
                            f"",
                            f"检测窗口:  第 {noise_ppo_update_count} 个 PPO 更新窗口",
                            f"涉及回合:  第 {window_start_ep} 轮 ~ 第 {window_end_ep} 轮",
                            f"上次窗口平均奖励:  {noise_prev_avg_reward[0]:.4f}",
                            f"本次窗口平均奖励:  {avg_raw_final_reward:.4f}",
                            f"下降幅度:  {reward_drop:.4f}  （告警阈值: {REWARD_DROP_WARNING_THRESHOLD:.2f}）",
                            f"",
                            f"请查看以下详情文件定位具体回合信息:",
                        ]
                        for _df in involved_basenames:
                            _warn_lines.append(f"  → details/{_df}")
                        _warn_lines.extend([
                            f"",
                            f"建议: 检查上述回合的终局奖励、约束余量和成本变化，",
                            f"      排查策略是否出现灾难性遗忘或探索越界。",
                        ])
                        _log_rounded_box(ev.log, _warn_lines)
                        # 同时立即追加写入警告报告文件
                        _write_warning_report(
                            noise_warning_file,
                            noise_warnings,
                            stage_label="噪声阶段（Stage-2 Noise）",
                        )
                noise_prev_avg_reward[0] = avg_raw_final_reward

                # [LEGACY] 窗口 top-k 候选确认已禁用，选择完全依赖 incumbent challenger 机制。
                # for _wti, _wtc in window_top_candidates:
                #     confirm_noise_candidate(
                #         _wtc,
                #         episode_idx=episode,
                #         window_idx=noise_ppo_update_count,
                #         repeats=NOISE_STAGE_CONFIRM_REPEATS,
                #         confirmation_label=f"window {noise_ppo_update_count + 1} top-{_wti+1}",
                #         update_shortlist=True,
                #     )
                window_best_score = float("-inf")
                window_best_cost = float("inf")
                window_best_noise_config = None
                window_top_candidates = []

                if (
                    USE_VALIDATION_FOR_REWARD
                    and (not use_fixed_search_reward)
                    and (episode + 1) < stage2_total_episodes
                ):
                    next_window_idx = noise_ppo_update_count
                    ev.refresh_validation_proxy(
                        window_index=next_window_idx,
                        stage_label="Stage-2 Noise RL",
                    )
                    online_reward_split = ev.get_online_reward_split_name()
                    rl_evaluator.split_name = online_reward_split
                    proxy_baseline_stats = ev.evaluate_model_with_attention_noise_repeated(
                        fixed_gelu,
                        fixed_softmax,
                        **baseline_noise_config,
                        repeats=NOISE_STAGE_ONLINE_BASELINE_REPEATS,
                        split=online_reward_split,
                    )
                    # 窗口轮换后同步更新 proxy baseline / limits / env 状态
                    proxy_base_loss = float(proxy_baseline_stats["loss_mean"])
                    proxy_base_p = float(proxy_baseline_stats["p_mean"])
                    proxy_base_s = float(proxy_baseline_stats["s_mean"])
                    proxy_worst_stats = ev.evaluate_model_with_attention_noise_repeated(
                        fixed_gelu,
                        fixed_softmax,
                        **worst_case_noise_config,
                        repeats=NOISE_STAGE_ONLINE_BASELINE_REPEATS,
                        split=online_reward_split,
                    )
                    proxy_limits = _compute_dynamic_limits(
                        proxy_base_loss, proxy_base_p, proxy_base_s,
                        float(proxy_worst_stats["loss_mean"]),
                        float(proxy_worst_stats["p_mean"]),
                        float(proxy_worst_stats["s_mean"]),
                    )
                    env.baseline_loss = proxy_base_loss
                    env.baseline_p = proxy_base_p
                    env.baseline_s = proxy_base_s
                    env.constraint_limits = proxy_limits
                    env.prev_episode_metrics = {
                        "loss": proxy_base_loss,
                        "metric1": proxy_base_p,
                        "metric2": proxy_base_s,
                        "cost": float(cost_reference_tot_c),
                    }
                    env.current_episode_metrics = None
                    ev.log(
                        f"  [窗口轮换] 代理子集（proxy）约束已刷新 → "
                        f"界限 limit(loss, m1, m2)=({proxy_limits['loss']:.4f}, "
                        f"{proxy_limits['metric1']:.4f}, "
                        f"{proxy_limits['metric2']:.4f})"
                    )

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
                    episode_returns,
                    episode_raw_final_rewards,
                    episode_losses,
                    episode_metric1s,
                    episode_metric2s,
                    episode_entropies,
                    base_loss=base_loss,
                    base_p=base_p,
                    base_s=base_s,
                    training_curve_path=progress_training_curve_path,
                    entropy_curve_path=progress_entropy_curve_path,
                    ppo_update_interval=PPO_UPDATE_INTERVAL,
                    use_validation=USE_VALIDATION_FOR_REWARD,
                )
                _plot_noise_risk_curves(
                    ev,
                    episode_safe_rates, episode_tail_violation_cvars, episode_tail_margin_scores,
                    confirm_window_indices, confirm_search_safe_rates, confirm_holdout_safe_rates,
                    confirm_search_tail_cvars, confirm_holdout_tail_cvars,
                    risk_curve_path=os.path.join(noise_progress_dir, f"noise_risk_curves_ep{episode + 1}.png"),
                    confirm_curve_path=os.path.join(noise_progress_dir, f"noise_confirm_curves_ep{episode + 1}.png"),
                )
                ev.log(
                    f"  [进度] 已写入训练/熵曲线快照 · 回合 episode={episode + 1}: "
                    f"{progress_training_curve_path}"
                )

        # [LEGACY] 训练结束后窗口 top-k 确认已禁用，选择完全依赖 incumbent challenger 机制。
        # if window_top_candidates:
        #     for _wti, _wtc in enumerate(window_top_candidates):
        #         confirm_noise_candidate(
        #             _wtc,
        #             episode_idx=stage2_total_episodes - 1,
        #             window_idx=noise_ppo_update_count,
        #             repeats=NOISE_STAGE_CONFIRM_REPEATS,
        #             confirmation_label=f"window {noise_ppo_update_count + 1} top-{_wti+1}",
        #             update_shortlist=True,
        #         )

        initial_shortlist_snapshot = []
        finalist_results = []
        finalist_best_noise_config = None
        # [LEGACY] Shortlist 二次确认/finalist 流程已禁用，选择完全依赖 incumbent challenger 机制。
        # if shortlist_candidates:
        #     ev.log("\n--- 噪声候选列表二次确认（Noise shortlist secondary confirmation） ---")
        #     for finalist_idx, shortlisted_candidate in enumerate(shortlist_candidates, start=1):
        #         finalist_candidate = confirm_noise_candidate(
        #             shortlisted_candidate,
        #             episode_idx=stage2_total_episodes - 1,
        #             window_idx=finalist_idx - 1,
        #             repeats=NOISE_STAGE_FINALIST_REPEATS,
        #             confirmation_label=f"finalist #{finalist_idx}",
        #             update_shortlist=False,
        #             is_finalist=True,
        #         )
        #         if finalist_candidate is None:
        #             continue
        #         finalist_results.append(_clone_candidate(finalist_candidate))
        #         if finalist_candidate["stable_joint_feasible"] and _is_better_joint_candidate(
        #             finalist_candidate,
        #             finalist_best_noise_config,
        #         ):
        #             finalist_best_noise_config = _clone_candidate(finalist_candidate)
        #             ev.log(
        #                 f"    最终稳定联合最优（Final Stable Joint-Best）从决赛候选（finalist）#{finalist_idx} 更新: "
        #                 f"成本（cost）={finalist_best_noise_config['cost']:.2f}, "
        #                 f"最终选择分数（final_selection_score）={finalist_best_noise_config['final_selection_score']:.4f}, "
        #                 f"稳定性分数（stability_score）={finalist_best_noise_config['stability_score']:.4f}"
        #             )

        status = NOISE_STAGE_STATUS_OK
        best_noise_config = (
            _clone_candidate(incumbent_best_noise_config)
            if incumbent_best_noise_config is not None
            and incumbent_best_noise_config.get("qualification_passed", False)
            else None
        )
        if best_noise_config is None:
            status = NOISE_STAGE_STATUS_NO_STABLE_FEASIBLE
            stable_search_best_noise_config = None
            stable_joint_best_noise_config = None
            ev.log(
                "\n  [结果] 在 val_search_full 与 val_holdout 上均未找到稳定可行的噪声方案"
                "（stable feasible noise config）。"
            )
        else:
            stable_search_best_noise_config = _clone_candidate(best_noise_config)
            stable_joint_best_noise_config = _clone_candidate(best_noise_config)

        _noise_log_major_rule(ev.log, "二阶段噪声 PPO 训练结束（Noise PPO training completed）")
        ev.log(
            f"  ▸ 阶段状态 stage status: "
            f"{_noise_stage_status_zh(status, NOISE_STAGE_STATUS_OK, NOISE_STAGE_STATUS_NO_STABLE_FEASIBLE)}"
        )
        if best_noise_config is not None:
            _noise_block_title(ev.log, "选定配置输出（Best stable noise configuration）")
            for key in (
                "input_noise_scaling_factors",
                "wq_noise_scaling_factors",
                "wk_noise_scaling_factors",
                "wv_noise_scaling_factors",
                "wo_noise_scaling_factors",
                "wffn1_noise_scaling_factors",
                "wffn2_noise_scaling_factors",
            ):
                ev.log(f"     {key}: {best_noise_config[key].tolist()}")
            ev.log(
                f"  ▸ 成本 cost={best_noise_config['cost']:.2f}  "
                f"│ 终选分 final_selection_score={best_noise_config['final_selection_score']:.4f}  "
                f"│ 原始终局 raw_final_reward={best_noise_config['raw_final_reward']:.4f}  "
                f"│ 稳定分 stability_score={best_noise_config['stability_score']:.4f}"
            )
        else:
            ev.log("  【输出】本次未选出稳定可行的噪声配置（no stable feasible selection）。")

        _plot_noise_training_curves(
            ev, episode_returns, episode_raw_final_rewards, episode_losses, episode_metric1s, episode_metric2s, episode_entropies,
            base_loss=base_loss, base_p=base_p, base_s=base_s,
            training_curve_path=noise_training_curve_path,
            entropy_curve_path=noise_entropy_curve_path,
            ppo_update_interval=PPO_UPDATE_INTERVAL,
            use_validation=USE_VALIDATION_FOR_REWARD,
        )
        noise_risk_curve_path = noise_training_curve_path.replace("training_curve", "risk_curves")
        noise_confirm_curve_path = noise_training_curve_path.replace("training_curve", "confirm_curves")
        _plot_noise_risk_curves(
            ev,
            episode_safe_rates, episode_tail_violation_cvars, episode_tail_margin_scores,
            confirm_window_indices, confirm_search_safe_rates, confirm_holdout_safe_rates,
            confirm_search_tail_cvars, confirm_holdout_tail_cvars,
            risk_curve_path=noise_risk_curve_path,
            confirm_curve_path=noise_confirm_curve_path,
        )

        if noise_step_info_chunk_file[0] is not None:
            noise_step_info_chunk_file[0].close()
            noise_step_info_chunk_file[0] = None

        if noise_warnings:
            # 最终报告已在每次警告触发时增量写入，此处仅做收尾提示
            ev.log(
                f"  ⚠ 训练期间共触发 {len(noise_warnings)} 条奖励骤降告警，"
                f"完整报告已保存至: {noise_warning_file}"
            )

        ev.total_episodes = original_total_episodes
        ev.apply_configuration(fixed_gelu, fixed_softmax)
        ev.clear_input_noise_configuration()
        ev.clear_weight_noise_configuration()

        reward_diagnostics = {
            "train_dense_eval_repeats": NOISE_STAGE_TRAIN_DENSE_EVAL_REPEATS,
            "train_terminal_eval_repeats": NOISE_STAGE_TRAIN_TERMINAL_EVAL_REPEATS,
            "dense_reward_mode": NOISE_STAGE_DENSE_REWARD_MODE,
            "train_terminal_reward_mode": NOISE_STAGE_TRAIN_TERMINAL_REWARD_MODE,
            "dense_completion_policy": NOISE_STAGE_DENSE_COMPLETION_POLICY,
            "mc_base_samples_legacy": NOISE_STAGE_MC_BASE_SAMPLES,
            "mc_extra_samples_legacy": NOISE_STAGE_MC_EXTRA_SAMPLES,
            "mc_margin_threshold": NOISE_STAGE_MC_MARGIN_THRESHOLD,
            "budget_decay_fraction": NOISE_STAGE_BUDGET_DECAY_FRACTION,
            "episode_return_mean": (
                float(np.mean(episode_returns))
                if episode_returns
                else None
            ),
            "final_selection_score_mean": (
                float(np.mean(episode_final_selection_scores))
                if episode_final_selection_scores
                else None
            ),
            "raw_final_reward_mean": (
                float(np.mean(episode_raw_final_rewards))
                if episode_raw_final_rewards
                else None
            ),
            "dense_reward_total_mean": (
                float(np.mean(episode_dense_reward_totals))
                if episode_dense_reward_totals
                else None
            ),
            "dense_reward_shaping_scale": float(NOISE_STAGE_DENSE_REWARD_SHAPING_SCALE),
            "stability_proxy_mean": (
                float(np.mean(stability_proxies))
                if stability_proxies
                else None
            ),
            "stability_penalty_mean": (
                float(np.mean(stability_penalties))
                if stability_penalties
                else None
            ),
        }

        selection_diagnostics = {
            "selection_mode": "incumbent_best_test",
            "legacy_shortlist_disabled": True,
            "best_test_repeats": int(NOISE_STAGE_BEST_TEST_REPEATS),
            "trigger_margin": float(NOISE_STAGE_BEST_TEST_TRIGGER_MARGIN),
            "safe_rate_min_static": float(NOISE_STAGE_BEST_TEST_SAFE_RATE_MIN),
            "cvar_max_static": float(NOISE_STAGE_BEST_TEST_CVAR_MAX),
            "dynamic_risk_thresholds": {
                split: {k: float(v) for k, v in thresholds.items()}
                for split, thresholds in dynamic_best_test_risk_thresholds.items()
            },
            "max_entry_score_drop": float(NOISE_STAGE_BEST_TEST_MAX_ENTRY_SCORE_DROP),
            "entry_std_multiplier": float(NOISE_STAGE_BEST_TEST_ENTRY_STD_MULTIPLIER),
            "qualification_trigger_count": int(qualification_trigger_count),
            "qualification_attempt_count": int(qualification_attempt_count),
            "qualification_reuse_count": int(qualification_reuse_count),
            "qualification_pass_count": int(qualification_pass_count),
            "qualification_fail_count": int(qualification_fail_count),
            "qualification_cache_size": int(len(qualification_cache)),
            "incumbent_history": [dict(item) for item in incumbent_history],
            "final_incumbent": _clone_candidate(incumbent_best_noise_config),
        }
        shortlist_diagnostics = dict(selection_diagnostics)

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
            "baseline_repeats": int(NOISE_STAGE_BASELINE_REPEATS),
            "online_baseline_repeats": int(NOISE_STAGE_ONLINE_BASELINE_REPEATS),
            "search_baseline_stats": _copy_repeat_summary(split_baseline_stats["search"]),
            "holdout_baseline_stats": _copy_repeat_summary(split_baseline_stats["holdout"]),
            "worst_reference_stats": _copy_repeat_summary(worst_reference_stats),
            "worst_holdout_stats": _copy_repeat_summary(worst_holdout_stats),
            "worst_case_noise_config": {k: v.copy() for k, v in worst_case_noise_config.items()},
            "limit_computation_method": "dynamic_quartile",
            "limit_quartile": float(NOISE_STAGE_DYNAMIC_LIMIT_QUARTILE),
            "search_limits": {k: float(v) for k, v in search_limits.items()},
            "holdout_limits": {k: float(v) for k, v in holdout_limits.items()},
            "status": status,
            "stability_thresholds": {
                split: dict(values)
                for split, values in stability_thresholds.items()
            },
            "best_noise_config": _clone_candidate(best_noise_config),
            "stable_search_best_noise_config": _clone_candidate(stable_search_best_noise_config),
            "stable_joint_best_noise_config": _clone_candidate(stable_joint_best_noise_config),
            "selection_diagnostics": selection_diagnostics,
            "shortlist_diagnostics": shortlist_diagnostics,
            "limit_loss": float(limit_loss),
            "limit_p": float(limit_p),
            "limit_s": float(limit_s),
            "proxy_limit_loss": float(proxy_limits["loss"]),
            "proxy_limit_p": float(proxy_limits["metric1"]),
            "proxy_limit_s": float(proxy_limits["metric2"]),
            "proxy_base_loss": float(proxy_base_loss),
            "proxy_base_p": float(proxy_base_p),
            "proxy_base_s": float(proxy_base_s),
            "training_eval_split": str(online_reward_split),
            "training_hparams": training_hparams,
            "reward_diagnostics": reward_diagnostics,
        }


# ---------------------------------------------------------------------------
# Internal helpers (module-private)
# ---------------------------------------------------------------------------

class _NoiseRecurrentRolloutBuffer:
    """Rollout buffer for the second-stage 7-action noise RL（支持双头 critic）。

    辅助 critic target 已从"终点 mean_perf"改为"每一步的 partial probe score"
    （文档修改方案 第6/7步）。字段名保持 mean_perf_targets 以兼容 PPO 接口。
    """

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
            "mean_perf_targets": [],  # 实际语义：每步 partial probe score（文档第6步）
        }

    def add_step(self, cont_feat, layer_idx, prev_actions, actions, logprob, reward, value, done,
                 mean_perf_target=0.0):
        self._current["cont_features"].append(cont_feat)
        self._current["layer_indices"].append(layer_idx)
        self._current["prev_actions"].append(prev_actions)
        self._current["actions"].append(actions)
        self._current["logprobs"].append(logprob)
        self._current["rewards"].append(reward)
        self._current["values"].append(value)
        self._current["dones"].append(done)
        # 语义变更：从"最后才冒出来的 mean_perf"改为"当前前缀 completion quality"（文档第6步）
        self._current["mean_perf_targets"].append(float(mean_perf_target))

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
            torch.tensor(ep["layer_indices"], dtype=torch.long) for ep in self.episodes
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

        mean_perf_targets = torch.tensor([
            ep["mean_perf_targets"] for ep in self.episodes
        ], dtype=torch.float32).to(device)

        return (cont_features, layer_indices, prev_actions, actions,
                logprobs, rewards, values, dones, mean_perf_targets)


class _NoiseGTrXLStrategyNetwork(nn.Module):
    """Second-stage GTrXL actor-critic with 7 independent noise-action heads."""

    action_names = ("x", "wq", "wk", "wv", "wo", "wffn1", "wffn2")

    def __init__(self, num_layers=12, d_model=64,
                 n_heads=4, n_gtrxl_layers=3,
                 d_ff=128, dropout=0.1,
                 gtrxl_block_cls=None,
                 lstm_pos_dim=16, lstm_proj_dim=32,
                 noise_stage_num_actions=7,
                 noise_stage_sos_tokens=None,
                 noise_stage_prev_action_embed_dim=4,
                 noise_stage_cont_dim=12,
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
            lstm_pos_dim +
            noise_stage_num_actions * noise_stage_prev_action_embed_dim +
            lstm_proj_dim
        )
        self.input_proj = nn.Identity() if token_input_dim == d_model else nn.Linear(token_input_dim, d_model)

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

        # 双头 critic：tail value head（主 critic）+ mean aux head（辅助监督）
        self.critic_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.mean_aux_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self._causal_mask_cache = {}
        self._initialize_weights()

    def _initialize_weights(self):
        for module in [self.actor_head, self.critic_head, self.mean_aux_head, self.fc_continuous]:
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
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
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
        tail_values = self.critic_head(x).squeeze(-1)
        mean_aux_values = self.mean_aux_head(x).squeeze(-1)
        return logits, tail_values, mean_aux_values

    def get_action_and_logprob(self, cont_features, layer_indices, prev_actions, return_probs=False):
        logits_dict, tail_values, _mean_aux_values = self.forward(cont_features, layer_indices, prev_actions)
        value = tail_values[:, -1].squeeze(0)

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
        logits_dict, tail_values, mean_aux_values = self.forward(cont_features, layer_indices, prev_actions)
        logprobs = torch.zeros_like(tail_values)
        entropy = torch.zeros_like(tail_values)
        for idx, name in enumerate(self.action_names):
            dist = Categorical(logits=logits_dict[name])
            logprobs = logprobs + dist.log_prob(actions[:, :, idx])
            entropy = entropy + dist.entropy()
        return logprobs, entropy, tail_values, mean_aux_values


class _NoiseOptEnv:
    """Second-stage RL environment over x/Wq/Wk/Wv/Wo/Wffn1/Wffn2 noise scaling factors."""

    def __init__(self, total_layers, baseline_cost, baseline_metrics, evaluator,
                 fixed_gelu, fixed_softmax, constraint_limits=None, prev_metrics=None, num_metrics=2,
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
                 reward_threshold=0.01, reward_dense_scale=0.1,
                 reward_cost_weight=20.0, reward_safety_bonus=1.0,
                 reward_clip_min=-5.0, reward_clip_max=5.0,
                 reward_normalization_scale=20.0,
                 budget_deviation_scale=0.05,
                 diff_reward_scale_acc=50.0, diff_reward_power=0.5,
                 log_barrier_violation_scale=10.0,
                 log_barrier_violation_steepness=20.0,
                 log_barrier_satisfaction_scale=0.5,
                 final_reward_alpha_perf=0.75,
                 final_reward_alpha_cost=0.25,
                 perf_weight_loss=0.15,
                 perf_weight_m1=0.425,
                 perf_weight_m2=0.425,
                 barrier_weight_loss=0.10,
                 barrier_weight_m1=0.45,
                 barrier_weight_m2=0.45,
                 mc_samples=5,
                 stability_weight=0.15,
                 stability_proxy_std_ref=0.008,
                 budget_decay_fraction=0.5,
                 mc_extra_samples=4,
                 mc_margin_threshold=0.02,
                 dense_reward_shaping_scale=0.25,
                 train_dense_eval_repeats=1,
                 train_terminal_eval_repeats=1,
                 dense_reward_mode="prefix_completion_delta",
                 train_terminal_reward_mode="single_shot_margin",
                 dense_completion_policy="max_suffix"):
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
            wffn1_noise_allowed
            if wffn1_noise_allowed is not None
            else weight_noise_allowed
        )
        self._input_noise_cost_map = input_noise_cost_map
        self._weight_noise_cost_map = weight_noise_cost_map
        self._wffn1_noise_cost_map = (
            wffn1_noise_cost_map
            if wffn1_noise_cost_map is not None
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
            wffn1_noise_scaling_to_norm
            if wffn1_noise_scaling_to_norm is not None
            else weight_noise_scaling_to_norm
        )
        self._sos_tokens = noise_stage_sos_tokens
        self._num_actions = noise_stage_num_actions
        self._history_mask = history_mask_value
        self._reward_threshold = reward_threshold
        self._reward_dense_scale = reward_dense_scale
        self._reward_cost_weight = reward_cost_weight
        self._reward_safety_bonus = reward_safety_bonus
        self._reward_clip_min = reward_clip_min
        self._reward_clip_max = reward_clip_max
        self._reward_norm_scale = reward_normalization_scale
        self._budget_dev_scale = budget_deviation_scale
        self._diff_reward_scale_acc = diff_reward_scale_acc
        self._diff_reward_power = diff_reward_power
        self._log_barrier_viol_scale = log_barrier_violation_scale
        self._log_barrier_viol_steep = log_barrier_violation_steepness
        self._log_barrier_sat_scale = log_barrier_satisfaction_scale
        self._final_reward_alpha_perf, self._final_reward_alpha_cost = (
            _validate_stage2_alpha_weights(
                final_reward_alpha_perf,
                final_reward_alpha_cost,
            )
        )
        # Legacy reward 所需的兼容属性（_assemble_final_reward_legacy 使用）
        self._reward_priority_performance = float(final_reward_alpha_perf)
        self._reward_priority_cost = float(final_reward_alpha_cost)
        self._cost_perf_align_scale = float(reward_safety_bonus)
        self._perf_weights = _resolve_stage2_metric_weights(
            num_metrics,
            {
                "loss": perf_weight_loss,
                "metric1": perf_weight_m1,
                "metric2": perf_weight_m2,
            },
            label="perf_weights",
        )
        self._barrier_weights = _resolve_stage2_metric_weights(
            num_metrics,
            {
                "loss": barrier_weight_loss,
                "metric1": barrier_weight_m1,
                "metric2": barrier_weight_m2,
            },
            label="barrier_weights",
        )
        self._mc_samples = max(1, int(mc_samples))
        self._stability_weight = float(stability_weight)
        self._stability_proxy_std_ref = float(stability_proxy_std_ref)
        self._budget_decay_fraction = max(0.01, float(budget_decay_fraction))
        self._episode_progress = 0.0
        self._mc_extra_samples = max(0, int(mc_extra_samples))
        self._mc_margin_threshold = float(mc_margin_threshold)
        self._dense_reward_shaping_scale = float(dense_reward_shaping_scale)

        # 训练期评估次数分离（文档修改方案 第1步）
        self._train_dense_eval_repeats = max(1, int(train_dense_eval_repeats))
        self._train_terminal_eval_repeats = max(1, int(train_terminal_eval_repeats))
        self._dense_reward_mode = str(dense_reward_mode)
        self._train_terminal_reward_mode = str(train_terminal_reward_mode)
        self._dense_completion_policy = str(dense_completion_policy)

        # dense probe 状态（文档修改方案 第2/3/5步）
        self._prev_completion_score = 0.0       # 上一步的 completion score，用于增量计算
        self._partial_probe_loss = 0.0
        self._partial_probe_metric1 = 0.0
        self._partial_probe_metric2 = 0.0
        self._partial_margin_loss = 0.0
        self._partial_margin_metric1 = 0.0
        self._partial_margin_metric2 = 0.0
        self._partial_completion_score = 0.0
        self._partial_violation_penalty = 0.0
        self._last_probe_info = None

        # 尾部风险状态（用于 continuous features 的后 3 维）
        self._prev_safe_rate = 1.0
        self._prev_tail_violation = 0.0
        self._prev_tail_margin = 0.0
        self._safe_rate_target = float(NOISE_STAGE_TRAIN_SAFE_RATE_TARGET)

        if constraint_limits is None:
            self.constraint_limits = {
                "loss": self.baseline_loss * (1 + self._reward_threshold),
                "metric1": self.baseline_p * (1 - self._reward_threshold),
                "metric2": self.baseline_s * (1 - self._reward_threshold),
            }
        else:
            self.constraint_limits = constraint_limits

        if prev_metrics is None:
            self.prev_episode_metrics = {
                "loss": self.baseline_loss,
                "metric1": self.baseline_p,
                "metric2": self.baseline_s,
                "cost": self.baseline_cost,
            }
        else:
            self.prev_episode_metrics = prev_metrics

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
        self.min_cost_per_layer = (
            min_input_cost
            + 5 * min_generic_weight_cost
            + min_wffn1_cost
        )
        self.max_cost_per_layer = (
            max_input_cost
            + 5 * max_generic_weight_cost
            + max_wffn1_cost
        )
        self.expected_cost_per_layer = (
            mean_input_cost
            + 5 * mean_generic_weight_cost
            + mean_wffn1_cost
        )
        self._cost_lower_bound = self.min_cost_per_layer * total_layers
        self._cost_upper_bound = float(self.baseline_cost)

        self.current_episode_metrics = None
        self.last_reward_components = None
        self.last_mc_eval = None
        self.reset()

    def reset(self):
        self.current_layer = 0
        self.accumulated_cost = 0.0
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

        # 重置 dense probe 状态（文档修改方案 第2/3步）
        self._prev_completion_score = 0.0
        self._partial_probe_loss = 0.0
        self._partial_probe_metric1 = 0.0
        self._partial_probe_metric2 = 0.0
        self._partial_margin_loss = 0.0
        self._partial_margin_metric1 = 0.0
        self._partial_margin_metric2 = 0.0
        self._partial_completion_score = 0.0
        self._partial_violation_penalty = 0.0
        self._last_probe_info = None

        self.accumulated_dense_reward = 0.0
        self.input_history = np.full(self.total_layers, self._history_mask, dtype=np.float32)
        self.wq_history = np.full(self.total_layers, self._history_mask, dtype=np.float32)
        self.wk_history = np.full(self.total_layers, self._history_mask, dtype=np.float32)
        self.wv_history = np.full(self.total_layers, self._history_mask, dtype=np.float32)
        self.wo_history = np.full(self.total_layers, self._history_mask, dtype=np.float32)
        self.wffn1_history = np.full(self.total_layers, self._history_mask, dtype=np.float32)
        self.wffn2_history = np.full(self.total_layers, self._history_mask, dtype=np.float32)
        return self._get_state()

    def _get_risk_features(self):
        """后 3 维风险预算特征（替代旧的均值预算特征）。

        safe_rate_gap 自然在 [-1, 1] 范围内，使用硬裁剪即可。
        tail_violation 和 tail_margin 可能显著超过 1.0（加权违约/余量之和），
        使用 tanh 软压缩保留大值区分度，避免硬裁剪丢失信息。
        """
        prev_safe_rate_gap = float(np.clip(
            self._safe_rate_target - self._prev_safe_rate, -1.0, 1.0
        ))
        prev_tail_violation = float(np.tanh(self._prev_tail_violation))
        prev_tail_margin = float(np.tanh(self._prev_tail_margin))
        return (prev_safe_rate_gap, prev_tail_violation, prev_tail_margin)

    def get_continuous_features(self):
        expected_cost_so_far = self.current_layer * self.expected_cost_per_layer
        if expected_cost_so_far > 0:
            cost_deviation = (self.accumulated_cost - expected_cost_so_far) / expected_cost_so_far
        else:
            cost_deviation = 0.0
        cost_deviation = np.clip(cost_deviation, -1.0, 1.0)

        baseline_cost_so_far = self.current_layer * self.max_cost_per_layer
        if baseline_cost_so_far > 0:
            complexity_debt = (baseline_cost_so_far - self.accumulated_cost) / baseline_cost_so_far
        else:
            complexity_debt = 0.0
        complexity_debt = np.clip(complexity_debt, 0.0, 1.0)

        progress = self.current_layer / self.total_layers
        safe_rate_gap, tail_violation, tail_margin = self._get_risk_features()

        # 当前 episode 的 partial probe 结果（文档修改方案 第5步）
        partial_completion_score = float(np.clip(self._partial_completion_score, -2.0, 2.0))
        partial_worst_margin = float(np.clip(
            min(self._partial_margin_loss, self._partial_margin_metric1,
                self._partial_margin_metric2 if self.num_metrics > 1 else self._partial_margin_metric1),
            -2.0, 2.0
        ))
        remaining_budget_ratio = float(np.clip(
            (float(self._cost_upper_bound) - float(self.accumulated_cost))
            / max(float(self._cost_upper_bound) - float(self._cost_lower_bound), 1e-8),
            0.0, 1.0,
        ))
        partial_margin_loss = float(np.clip(self._partial_margin_loss, -2.0, 2.0))
        partial_margin_metric1 = float(np.clip(self._partial_margin_metric1, -2.0, 2.0))
        partial_margin_metric2 = float(np.clip(self._partial_margin_metric2, -2.0, 2.0))

        return np.array(
            [cost_deviation, complexity_debt, progress,
             safe_rate_gap, tail_violation, tail_margin,
             # 以下 6 维为 partial probe 特征（文档修改方案 第5步）
             partial_completion_score, partial_worst_margin, remaining_budget_ratio,
             partial_margin_loss, partial_margin_metric1, partial_margin_metric2],
            dtype=np.float32,
        )

    def update_prev_metrics(self):
        if self.current_episode_metrics is not None:
            self.prev_episode_metrics = self.current_episode_metrics.copy()

    def set_episode_progress(self, episode, total_episodes):
        self._episode_progress = float(episode) / max(1, int(total_episodes))

    def _get_current_constraint_limits(self):
        m1_limit = float(self.constraint_limits["metric1"])
        base_limits = {
            "loss": float(self.constraint_limits["loss"]),
            "metric1": m1_limit,
            "metric2": float(self.constraint_limits.get("metric2", m1_limit)),
        }
        base_evaluator = getattr(self.evaluator, "evaluator", None)
        if base_evaluator is not None and hasattr(base_evaluator, "get_curriculum_constraints"):
            return base_evaluator.get_curriculum_constraints(base_limits)
        return base_limits

    # -------------------------------------------------------------------
    # 第一类：训练期单次评估 helper（文档修改方案 第2步）
    # -------------------------------------------------------------------

    def _evaluate_noise_config_single(self, noise_kwargs=None):
        """训练期单次评估：返回一条 trial 风格结果 + n=1 的 summary，方便日志兼容。"""
        if noise_kwargs is None:
            noise_kwargs = {
                "input_noise_scaling_factors": np.array(self.input_noise_config, dtype=int),
                "wq_noise_scaling_factors": np.array(self.wq_noise_config, dtype=int),
                "wk_noise_scaling_factors": np.array(self.wk_noise_config, dtype=int),
                "wv_noise_scaling_factors": np.array(self.wv_noise_config, dtype=int),
                "wo_noise_scaling_factors": np.array(self.wo_noise_config, dtype=int),
                "wffn1_noise_scaling_factors": np.array(self.wffn1_noise_config, dtype=int),
                "wffn2_noise_scaling_factors": np.array(self.wffn2_noise_config, dtype=int),
            }
        loss, metric1, metric2, eval_time = self.evaluator.evaluate_noise_model(**noise_kwargs)
        trial = {
            "loss": float(loss),
            "metric1": float(metric1),
            "metric2": float(metric2),
            "time_ms": float(eval_time),
        }
        summary = {
            "num_samples": 1,
            "base_samples": 1,
            "extra_samples": 0,
            "trials": [trial],
            "loss_mean": float(loss),
            "loss_std": 0.0,
            "metric1_mean": float(metric1),
            "metric1_std": 0.0,
            "metric2_mean": float(metric2),
            "metric2_std": 0.0,
            "time_mean_ms": float(eval_time),
            "time_std_ms": 0.0,
        }
        return trial, summary

    # -------------------------------------------------------------------
    # 第二类：中间步 dense probe helper（文档修改方案 第2步）
    # -------------------------------------------------------------------

    def _compose_dense_probe_noise_kwargs(self):
        """把当前 episode 的前缀配置拼成完整 noise config（前缀部分已选，后缀用 completion 补齐）。"""
        prefix_kwargs = {
            "input_noise_scaling_factors": list(self.input_noise_config),
            "wq_noise_scaling_factors": list(self.wq_noise_config),
            "wk_noise_scaling_factors": list(self.wk_noise_config),
            "wv_noise_scaling_factors": list(self.wv_noise_config),
            "wo_noise_scaling_factors": list(self.wo_noise_config),
            "wffn1_noise_scaling_factors": list(self.wffn1_noise_config),
            "wffn2_noise_scaling_factors": list(self.wffn2_noise_config),
        }
        # 用 completion policy 补齐未来层
        suffix = self._build_dense_completion_suffix(len(self.input_noise_config))
        for key in prefix_kwargs:
            prefix_kwargs[key] = np.array(
                prefix_kwargs[key] + suffix[key], dtype=int
            )
        return prefix_kwargs

    def _build_dense_completion_suffix(self, start_layer):
        """给尚未选择的未来层补上 completion 配置。

        当前策略: max_suffix — 未来层全部使用最大 scaling factor（最低噪声/最保守）。
        """
        remaining = self.total_layers - start_layer
        if remaining <= 0:
            return {
                "input_noise_scaling_factors": [],
                "wq_noise_scaling_factors": [],
                "wk_noise_scaling_factors": [],
                "wv_noise_scaling_factors": [],
                "wo_noise_scaling_factors": [],
                "wffn1_noise_scaling_factors": [],
                "wffn2_noise_scaling_factors": [],
            }
        max_input = max(self._input_noise_allowed)
        max_weight = max(self._weight_noise_allowed)
        max_wffn1 = max(self._wffn1_noise_allowed)
        return {
            "input_noise_scaling_factors": [max_input] * remaining,
            "wq_noise_scaling_factors": [max_weight] * remaining,
            "wk_noise_scaling_factors": [max_weight] * remaining,
            "wv_noise_scaling_factors": [max_weight] * remaining,
            "wo_noise_scaling_factors": [max_weight] * remaining,
            "wffn1_noise_scaling_factors": [max_wffn1] * remaining,
            "wffn2_noise_scaling_factors": [max_weight] * remaining,
        }

    def _evaluate_dense_probe_once(self):
        """对"前缀 + completion 后的完整 config"做 1 次 evaluator 调用，返回 trial 和 probe score。"""
        probe_kwargs = self._compose_dense_probe_noise_kwargs()
        loss, metric1, metric2, eval_time = self.evaluator.evaluate_noise_model(**probe_kwargs)
        trial = {
            "loss": float(loss),
            "metric1": float(metric1),
            "metric2": float(metric2),
            "time_ms": float(eval_time),
        }

        # 计算约束归一化后的 margin（文档修改方案 第3步 S(complete(prefix_t)) 里的内容）
        # 使用 abs() + 合理下限防止 limit 在 baseline 同侧时分母极小导致 margin 爆炸
        limits = self._get_current_constraint_limits()
        _MARGIN_DENOM_FLOOR = 0.01
        margin_loss = (float(limits["loss"]) - float(loss)) / max(
            abs(float(limits["loss"]) - float(self.baseline_loss)), _MARGIN_DENOM_FLOOR
        )
        margin_m1 = (float(metric1) - float(limits["metric1"])) / max(
            abs(float(self.baseline_p) - float(limits["metric1"])), _MARGIN_DENOM_FLOOR
        )
        if self.num_metrics > 1:
            margin_m2 = (float(metric2) - float(limits["metric2"])) / max(
                abs(float(self.baseline_s) - float(limits["metric2"])), _MARGIN_DENOM_FLOOR
            )
        else:
            margin_m2 = 0.0

        # violation penalty（加权违约量）
        viol_loss = max(0.0, -margin_loss)
        viol_m1 = max(0.0, -margin_m1)
        viol_m2 = max(0.0, -margin_m2) if self.num_metrics > 1 else 0.0
        violation_penalty = (
            float(self._barrier_weights["loss"]) * viol_loss
            + float(self._barrier_weights["metric1"]) * viol_m1
        )
        if self.num_metrics > 1:
            violation_penalty += float(self._barrier_weights.get("metric2", 0.0)) * viol_m2

        # completion score = 加权 margin utility - violation penalty + 轻量 cost 项
        util_loss = max(0.0, margin_loss)
        util_m1 = max(0.0, margin_m1)
        util_m2 = max(0.0, margin_m2) if self.num_metrics > 1 else 0.0
        margin_utility = (
            float(self._perf_weights["loss"]) * util_loss
            + float(self._perf_weights["metric1"]) * util_m1
        )
        if self.num_metrics > 1:
            margin_utility += float(self._perf_weights.get("metric2", 0.0)) * util_m2

        # 轻量 cost 项
        cost_score = float(np.clip(
            (float(self._cost_upper_bound) - float(self.accumulated_cost))
            / max(float(self._cost_upper_bound) - float(self._cost_lower_bound), 1e-8),
            0.0, 1.0,
        ))
        completion_score = margin_utility - violation_penalty + NOISE_STAGE_COST_WEIGHT * cost_score

        # 更新 partial probe 状态（文档修改方案 第5步：喂回 state）
        self._partial_probe_loss = float(loss)
        self._partial_probe_metric1 = float(metric1)
        self._partial_probe_metric2 = float(metric2)
        self._partial_margin_loss = float(margin_loss)
        self._partial_margin_metric1 = float(margin_m1)
        self._partial_margin_metric2 = float(margin_m2)
        self._partial_completion_score = float(completion_score)
        self._partial_violation_penalty = float(violation_penalty)

        probe_info = {
            "probe_trial": trial,
            "margin_loss": float(margin_loss),
            "margin_metric1": float(margin_m1),
            "margin_metric2": float(margin_m2),
            "violation_penalty": float(violation_penalty),
            "margin_utility": float(margin_utility),
            "cost_score": float(cost_score),
            "completion_score": float(completion_score),
            "completion_policy": self._dense_completion_policy,
        }
        return trial, float(completion_score), probe_info

    def _evaluate_noise_config_mc(self):
        """MC 采样评估：返回 trial 级数组 + summary stats，支持后续 tail 指标计算。"""
        trials = []
        noise_kwargs = {
            "input_noise_scaling_factors": np.array(self.input_noise_config, dtype=int),
            "wq_noise_scaling_factors": np.array(self.wq_noise_config, dtype=int),
            "wk_noise_scaling_factors": np.array(self.wk_noise_config, dtype=int),
            "wv_noise_scaling_factors": np.array(self.wv_noise_config, dtype=int),
            "wo_noise_scaling_factors": np.array(self.wo_noise_config, dtype=int),
            "wffn1_noise_scaling_factors": np.array(self.wffn1_noise_config, dtype=int),
            "wffn2_noise_scaling_factors": np.array(self.wffn2_noise_config, dtype=int),
        }

        for _ in range(self._mc_samples):
            loss, metric1, metric2, eval_time = self.evaluator.evaluate_noise_model(**noise_kwargs)
            trials.append({
                "loss": float(loss),
                "metric1": float(metric1),
                "metric2": float(metric2),
                "time_ms": float(eval_time),
            })

        losses = [t["loss"] for t in trials]
        metric1s = [t["metric1"] for t in trials]
        metric2s = [t["metric2"] for t in trials]
        times = [t["time_ms"] for t in trials]
        total_samples = len(trials)
        return {
            "num_samples": total_samples,
            "base_samples": self._mc_samples,
            "extra_samples": 0,
            "trials": trials,
            "loss_mean": float(np.mean(losses)),
            "loss_std": float(np.std(losses)),
            "metric1_mean": float(np.mean(metric1s)),
            "metric1_std": float(np.std(metric1s)),
            "metric2_mean": float(np.mean(metric2s)),
            "metric2_std": float(np.std(metric2s)),
            "time_mean_ms": float(np.mean(times)),
            "time_std_ms": float(np.std(times)),
        }

    def _assemble_final_reward_legacy(self, loss, m1, m2, mc_eval=None):
        limits = self._get_current_constraint_limits()
        r_diff = 0.0

        def violation_penalty(curr_value, limit_value, is_upper_bound=True):
            margin = (limit_value - curr_value) if is_upper_bound else (curr_value - limit_value)
            if margin < 0:
                penalty = -self._log_barrier_viol_scale * np.exp(-margin * self._log_barrier_viol_steep)
            else:
                penalty = 0.0
            return penalty, margin

        r_loss_barrier, margin_loss = violation_penalty(loss, limits["loss"], is_upper_bound=True)
        r_m1_barrier, margin_m1 = violation_penalty(m1, limits["metric1"], is_upper_bound=False)
        r_m2_barrier, margin_m2 = violation_penalty(m2, limits["metric2"], is_upper_bound=False)

        # Normalize margins by their respective constraint limits so that
        # loss, metric1, metric2 contribute equally to r_safe regardless of
        # their natural magnitude differences.
        norm_margin_loss = max(0.0, margin_loss) / max(abs(limits["loss"]), 1e-8)
        norm_margin_m1 = max(0.0, margin_m1) / max(abs(limits["metric1"]), 1e-8)
        norm_margin_m2 = max(0.0, margin_m2) / max(abs(limits["metric2"]), 1e-8)

        if self.num_metrics == 1:
            r_barrier = (r_loss_barrier + r_m1_barrier) / 2.0
            normalized_positive_margins = [norm_margin_loss, norm_margin_m1]
            constraints_ok = (margin_loss >= 0) and (margin_m1 >= 0)
        else:
            r_barrier = (r_loss_barrier + r_m1_barrier + r_m2_barrier) / 3.0
            normalized_positive_margins = [norm_margin_loss, norm_margin_m1, norm_margin_m2]
            constraints_ok = (margin_loss >= 0) and (margin_m1 >= 0) and (margin_m2 >= 0)

        cost_saving = (self.baseline_cost - self.accumulated_cost) / (self.baseline_cost + 1e-8)
        # Use auto-calibrated scale so r_cost ≈ safety_bonus at reference cost saving.
        # This ensures priority weights directly control the performance:cost ratio.
        r_cost = cost_saving * self._cost_perf_align_scale
        r_safe = 0.0
        if constraints_ok:
            r_safe = self._reward_safety_bonus + self._log_barrier_sat_scale * float(np.mean(normalized_positive_margins))

        performance_term = r_barrier + r_safe
        weighted_performance = self._reward_priority_performance * performance_term
        weighted_cost = self._reward_priority_cost * r_cost

        base_reward = (weighted_cost + weighted_performance) / self._reward_norm_scale

        # Stability penalty: lightweight linear penalty from MC evaluation variance
        stability_proxy = 0.0
        stability_penalty = 0.0
        if mc_eval is not None:
            std_components = [
                float(mc_eval.get("loss_std", 0.0)) / self._stability_proxy_std_ref,
                float(mc_eval.get("metric1_std", 0.0)) / self._stability_proxy_std_ref,
            ]
            if self.num_metrics > 1:
                std_components.append(
                    float(mc_eval.get("metric2_std", 0.0)) / self._stability_proxy_std_ref
                )
            stability_proxy = float(np.mean(std_components)) if std_components else 0.0
            stability_penalty = -self._stability_weight * max(0.0, stability_proxy - 1.0)

        reward = np.clip(
            base_reward + stability_penalty,
            self._reward_clip_min,
            self._reward_clip_max,
        )
        reward_components = {
            "cost_saving": float(cost_saving),
            "cost": float(r_cost),
            "cost_perf_align_scale": float(self._cost_perf_align_scale),
            "diff": float(r_diff),
            "barrier": float(r_barrier),
            "safety": float(r_safe),
            "performance_term": float(performance_term),
            "weighted_performance": float(weighted_performance),
            "weighted_cost": float(weighted_cost),
            "perf_cost_contribution_ratio": (
                float(weighted_performance / weighted_cost) if abs(weighted_cost) > 1e-8 else float("inf")
            ),
            "reward_priority_performance": float(self._reward_priority_performance),
            "reward_priority_cost": float(self._reward_priority_cost),
            "constraints_ok": bool(constraints_ok),
            "cost_reward_active": True,
            "loss_limit": float(limits["loss"]),
            "metric1_limit": float(limits["metric1"]),
            "metric2_limit": float(limits["metric2"]),
            "margin_loss": float(margin_loss),
            "margin_metric1": float(margin_m1),
            "margin_metric2": float(margin_m2),
            "stability_proxy": float(stability_proxy),
            "stability_penalty": float(stability_penalty),
            "stability_weight": float(self._stability_weight),
            "raw_final_reward": float(reward),
        }
        self.last_cost_reward = float(weighted_cost / self._reward_norm_scale)
        self.last_acc_reward = float(weighted_performance / self._reward_norm_scale)
        self.last_reward_components = reward_components
        return float(reward), reward_components

    def score_reward_components(self, loss, m1, m2, cost, mc_eval=None, constraint_limits=None):
        limits = (
            constraint_limits
            if constraint_limits is not None
            else self._get_current_constraint_limits()
        )
        return _compute_stage2_reward_components(
            loss=loss,
            metric1=m1,
            metric2=m2,
            baseline_metrics=(self.baseline_loss, self.baseline_p, self.baseline_s),
            constraint_limits=limits,
            cost_value=cost,
            cost_lower_bound=self._cost_lower_bound,
            cost_upper_bound=self._cost_upper_bound,
            final_reward_alpha_perf=self._final_reward_alpha_perf,
            final_reward_alpha_cost=self._final_reward_alpha_cost,
            perf_weights=self._perf_weights,
            barrier_weights=self._barrier_weights,
            stability_proxy_std_ref=self._stability_proxy_std_ref,
            stability_weight=self._stability_weight,
            num_metrics=self.num_metrics,
            mc_eval=mc_eval,
        )

    def _assemble_final_reward(self, loss, m1, m2, mc_eval=None):
        reward, reward_components = self.score_reward_components(
            loss,
            m1,
            m2,
            self.accumulated_cost,
            mc_eval=mc_eval,
        )
        self.last_cost_reward = float(
            self._final_reward_alpha_cost * reward_components["cost_score"]
        )
        self.last_acc_reward = float(
            self._final_reward_alpha_perf * reward_components["perf_score"]
        )
        self.last_reward_components = dict(reward_components)
        return float(reward), reward_components

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

    def _compute_dense_step_reward(self, step_cost):
        """prefix-completion 增量 dense reward（文档修改方案 第3步）。

        r_t^dense = S(complete(prefix_t)) - S(complete(prefix_{t-1}))
        当前层只为自己"让完成态变好/变坏了多少"负责，credit 更干净。
        """
        # 执行 dense probe 评估
        _trial, current_score, probe_info = self._evaluate_dense_probe_once()

        # 增量形式
        delta = current_score - self._prev_completion_score
        self._prev_completion_score = current_score

        dense_reward = self._dense_reward_shaping_scale * delta
        # 将 probe_info 暂存，供 step() 回传 info
        self._last_probe_info = probe_info
        return dense_reward

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
            self._input_noise_cost_map[input_sf] +
            self._weight_noise_cost_map[wq_sf] +
            self._weight_noise_cost_map[wk_sf] +
            self._weight_noise_cost_map[wv_sf] +
            self._weight_noise_cost_map[wo_sf] +
            self._wffn1_noise_cost_map[wffn1_sf] +
            self._weight_noise_cost_map[wffn2_sf]
        )
        self.accumulated_cost += step_cost

        self.prev_action_indices = np.array([
            int(input_action_idx), int(wq_action_idx), int(wk_action_idx), int(wv_action_idx),
            int(wo_action_idx), int(wffn1_action_idx), int(wffn2_action_idx)
        ], dtype=np.int64)
        self.prev_scalings = {
            "x": input_sf,
            "wq": wq_sf,
            "wk": wk_sf,
            "wv": wv_sf,
            "wo": wo_sf,
            "wffn1": wffn1_sf,
            "wffn2": wffn2_sf,
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
            # 非终止步：返回 prefix-completion 增量 dense reward（文档修改方案 第3步）
            dense_reward = self._compute_dense_step_reward(step_cost)
            self.accumulated_dense_reward += dense_reward
            info["dense_reward"] = dense_reward
            info["accumulated_dense_reward"] = self.accumulated_dense_reward
            info["dense_reward_adjustment"] = dense_reward
            info["dense_reward_cancelled"] = False
            # dense probe 详情（文档修改方案 第8步：日志扩展）
            probe_info = getattr(self, "_last_probe_info", None) or {}
            info["partial_probe_loss"] = probe_info.get("probe_trial", {}).get("loss")
            info["partial_probe_metric1"] = probe_info.get("probe_trial", {}).get("metric1")
            info["partial_probe_metric2"] = probe_info.get("probe_trial", {}).get("metric2")
            info["partial_margin_loss"] = probe_info.get("margin_loss")
            info["partial_margin_metric1"] = probe_info.get("margin_metric1")
            info["partial_margin_metric2"] = probe_info.get("margin_metric2")
            info["partial_completion_score"] = probe_info.get("completion_score")
            info["partial_violation_penalty"] = probe_info.get("violation_penalty")
            info["remaining_budget_ratio"] = float(np.clip(
                (float(self._cost_upper_bound) - float(self.accumulated_cost))
                / max(float(self._cost_upper_bound) - float(self._cost_lower_bound), 1e-8),
                0.0, 1.0,
            ))
            info["completion_policy"] = self._dense_completion_policy
            # 辅助 critic target：每一步的 partial probe score（文档修改方案 第6步）
            info["mean_perf_value"] = float(self._partial_completion_score)
            return self._get_state(), dense_reward, False, info

        # 终止步：只返回 terminal sparse reward，不再叠加 dense delta（文档修改方案 第3步 大坑修正）
        dense_reward = 0.0  # 最后一步不叠加 dense delta
        info["dense_reward"] = dense_reward
        final_reward = self._compute_final_reward()
        rc = final_reward["reward_components"]
        info["final_reward"] = final_reward["raw_final_reward"]
        info["raw_final_reward"] = final_reward["raw_final_reward"]
        info["final_selection_score"] = rc["final_selection_score"]
        info["mc_eval"] = final_reward["mc_eval"]
        info["reward_components"] = rc
        info["accumulated_dense_reward"] = self.accumulated_dense_reward
        info["dense_reward_adjustment"] = dense_reward
        info["dense_reward_cancelled"] = False
        # 尾部安全指标
        info["train_safe_rate"] = rc.get("safe_rate", 0.0)
        info["train_tail_k"] = rc.get("tail_k", 0)
        info["train_tail_violation_cvar"] = rc.get("tail_violation_cvar", 0.0)
        info["train_tail_margin_score"] = rc.get("tail_margin_score", 0.0)
        info["train_mean_perf_score"] = rc.get("mean_perf_score", 0.0)
        info["train_cost_score"] = rc.get("cost_score", 0.0)
        info["raw_tail_reward"] = rc.get("raw_tail_reward", 0.0)
        info["unsafe_sample_count"] = rc.get("unsafe_count", 0)
        # 辅助 critic target：终点用 completion score（文档修改方案 第6步）
        info["mean_perf_value"] = float(self._partial_completion_score)
        # 终止步只返回 terminal sparse reward
        terminal_reward = final_reward["raw_final_reward"]
        info["total_reward"] = terminal_reward
        return self._get_state(), terminal_reward, True, info

    def _compute_final_reward(self):
        """训练期 terminal reward：使用单次评估 + single_shot_margin（文档修改方案 第4步）。

        训练时学一个"单样本可学习"的 terminal objective，
        评估/筛选时仍然用 repeated tail-safe objective（在 confirm/finalist 流程中）。
        """
        # 训练期使用单次评估（文档修改方案 第1/4步）
        trial, mc_eval = self._evaluate_noise_config_single()
        loss = trial["loss"]
        m1 = trial["metric1"]
        m2 = trial["metric2"]

        self.current_episode_metrics = {
            "loss": loss,
            "metric1": m1,
            "metric2": m2,
            "cost": self.accumulated_cost,
        }

        # 使用 single_shot_margin 模式计算 terminal reward（文档修改方案 第4步）
        raw_final_reward, reward_components = self._compute_single_shot_terminal_reward(
            loss, m1, m2, mc_eval=mc_eval,
        )
        self.last_reward_components = dict(reward_components)
        self.last_mc_eval = mc_eval

        # 更新尾部风险状态（供下一 episode 的 continuous features 使用）
        self._prev_safe_rate = reward_components.get("safe_rate", 1.0)
        self._prev_tail_violation = reward_components.get("tail_violation_cvar", 0.0)
        self._prev_tail_margin = reward_components.get("tail_margin_score", 0.0)

        return {
            "raw_final_reward": raw_final_reward,
            "mc_eval": mc_eval,
            "reward_components": reward_components,
        }

    def _compute_single_shot_terminal_reward(self, loss, m1, m2, mc_eval=None):
        """训练期 single-shot terminal reward：基于单次评估的归一化 margin + violation penalty + 轻量 cost score。

        不再使用 tail-safe 多试验 reward，因为 n=1 时 tail 概念不成立（文档修改方案 第4步）。
        """
        limits = self._get_current_constraint_limits()
        raw_reward, reward_components = _compute_stage2_single_shot_reward_components(
            loss=loss,
            metric1=m1,
            metric2=m2,
            baseline_metrics=(self.baseline_loss, self.baseline_p, self.baseline_s),
            constraint_limits=limits,
            cost_value=self.accumulated_cost,
            cost_lower_bound=self._cost_lower_bound,
            cost_upper_bound=self._cost_upper_bound,
            perf_weights=self._perf_weights,
            barrier_weights=self._barrier_weights,
            num_metrics=self.num_metrics,
            safe_rate_target=self._safe_rate_target,
        )
        # 安全裁剪：防止极端 reward 摧毁策略网络（与 legacy 路径对齐）
        clipped_reward = float(np.clip(
            raw_reward, self._reward_clip_min, self._reward_clip_max
        ))
        if clipped_reward != raw_reward:
            reward_components["reward_clipped"] = True
            reward_components["raw_reward_before_clip"] = float(raw_reward)
        self.last_cost_reward = float(
            self._final_reward_alpha_cost * reward_components["cost_score"]
        )
        self.last_acc_reward = float(
            self._final_reward_alpha_perf * reward_components["perf_score"]
        )
        return clipped_reward, reward_components

        # ===================================================================
        # [LEGACY] 以下为旧版内联 single-shot terminal reward 实现。
        # 已由上方 _compute_stage2_single_shot_reward_components() 调用替代。
        # 位于 return 语句之后，为不可达代码，保留供参考。
        # ===================================================================
        # # 归一化 margin
        # margin_loss = (float(limits["loss"]) - float(loss)) / max(
        #     float(limits["loss"]) - float(self.baseline_loss), 1e-8
        # )
        # margin_m1 = (float(m1) - float(limits["metric1"])) / max(
        #     float(self.baseline_p) - float(limits["metric1"]), 1e-8
        # )
        # if self.num_metrics > 1:
        #     margin_m2 = (float(m2) - float(limits["metric2"])) / max(
        #         float(self.baseline_s) - float(limits["metric2"]), 1e-8
        #     )
        # else:
        #     margin_m2 = 0.0
        #
        # # violation penalty
        # viol_loss = max(0.0, -margin_loss)
        # viol_m1 = max(0.0, -margin_m1)
        # viol_m2 = max(0.0, -margin_m2) if self.num_metrics > 1 else 0.0
        # violation_penalty = (
        #     float(self._barrier_weights["loss"]) * viol_loss
        #     + float(self._barrier_weights["metric1"]) * viol_m1
        # )
        # if self.num_metrics > 1:
        #     violation_penalty += float(self._barrier_weights.get("metric2", 0.0)) * viol_m2
        #
        # # margin utility
        # util_loss = max(0.0, margin_loss)
        # util_m1 = max(0.0, margin_m1)
        # util_m2 = max(0.0, margin_m2) if self.num_metrics > 1 else 0.0
        # margin_utility = (
        #     float(self._perf_weights["loss"]) * util_loss
        #     + float(self._perf_weights["metric1"]) * util_m1
        # )
        # if self.num_metrics > 1:
        #     margin_utility += float(self._perf_weights.get("metric2", 0.0)) * util_m2
        #
        # # 轻量 cost score
        # cost_score = float(np.clip(
        #     (float(self._cost_upper_bound) - float(self.accumulated_cost))
        #     / max(float(self._cost_upper_bound) - float(self._cost_lower_bound), 1e-8),
        #     0.0, 1.0,
        # ))
        # cost_saving = (float(self._cost_upper_bound) - float(self.accumulated_cost)) / max(
        #     float(self._cost_upper_bound), 1e-8
        # )
        #
        # # 约束通过判定
        # constraints_ok = (margin_loss >= 0.0) and (margin_m1 >= 0.0)
        # if self.num_metrics > 1:
        #     constraints_ok = constraints_ok and (margin_m2 >= 0.0)
        #
        # # safe_rate / safe_rate_gap 的单样本近似
        # safe_rate = 1.0 if constraints_ok else 0.0
        # safe_rate_gap = max(0.0, float(self._safe_rate_target) - safe_rate)
        #
        # # single-shot terminal reward（与 R_tail-train 结构对齐，但基于单样本）
        # raw_reward = (
        #     float(NOISE_STAGE_TAIL_MARGIN_WEIGHT) * margin_utility
        #     - float(NOISE_STAGE_TAIL_VIOLATION_WEIGHT) * violation_penalty
        #     - float(NOISE_STAGE_SAFE_RATE_GAP_WEIGHT) * safe_rate_gap
        #     + float(NOISE_STAGE_MEAN_PERF_WEIGHT) * margin_utility  # mean_perf 退化为 margin_utility
        #     + float(NOISE_STAGE_COST_WEIGHT) * cost_score
        # )
        #
        # reward_components = {
        #     "loss_limit": float(limits["loss"]),
        #     "metric1_limit": float(limits["metric1"]),
        #     "metric2_limit": float(limits.get("metric2", limits["metric1"])),
        #     "loss_ratio": float(margin_loss),
        #     "metric1_ratio": float(margin_m1),
        #     "metric2_ratio": float(margin_m2),
        #     "perf_score": float(margin_utility),
        #     "cost_score": float(cost_score),
        #     "barrier_penalty": float(violation_penalty),
        #     "stability_proxy": 0.0,
        #     "stability_penalty": 0.0,
        #     "constraints_ok": bool(constraints_ok),
        #     "cost_lower_bound": float(self._cost_lower_bound),
        #     "cost_upper_bound": float(self._cost_upper_bound),
        #     "current_cost": float(self.accumulated_cost),
        #     "cost_saving": float(cost_saving),
        #     "metric2_active": bool(self.num_metrics > 1),
        #     # 单样本风险指标
        #     "safe_rate": float(safe_rate),
        #     "safe_rate_gap": float(safe_rate_gap),
        #     "tail_k": 1,
        #     "tail_violation_cvar": float(violation_penalty),
        #     "tail_margin_score": float(margin_utility),
        #     "mean_perf_score": float(margin_utility),
        #     "unsafe_count": 0 if constraints_ok else 1,
        #     "tail_loss_mean": float(loss),
        #     "tail_acc_mean": float(m1),
        #     "tail_f1_mean": float(m2) if self.num_metrics > 1 else float(m1),
        #     # reward 权重
        #     "tail_margin_weight": float(NOISE_STAGE_TAIL_MARGIN_WEIGHT),
        #     "tail_violation_weight": float(NOISE_STAGE_TAIL_VIOLATION_WEIGHT),
        #     "safe_rate_gap_weight": float(NOISE_STAGE_SAFE_RATE_GAP_WEIGHT),
        #     "mean_perf_weight_coef": float(NOISE_STAGE_MEAN_PERF_WEIGHT),
        #     "cost_weight_coef": float(NOISE_STAGE_COST_WEIGHT),
        #     "safe_rate_target": float(self._safe_rate_target),
        #     "raw_tail_reward": float(raw_reward),
        #     "raw_final_reward": float(raw_reward),
        #     "final_selection_score": float(raw_reward),
        #     "terminal_reward_mode": "single_shot_margin",
        # }
        # self.last_cost_reward = float(self._final_reward_alpha_cost * cost_score)
        # self.last_acc_reward = float(self._final_reward_alpha_perf * margin_utility)
        # return float(raw_reward), reward_components


# ---------------------------------------------------------------------------
# Module-private helper functions
# ---------------------------------------------------------------------------

def _validate_stage2_alpha_weights(alpha_perf, alpha_cost, tol=NOISE_STAGE_WEIGHT_TOL):
    alpha_perf = float(alpha_perf)
    alpha_cost = float(alpha_cost)
    if not np.isclose(alpha_perf + alpha_cost, 1.0, atol=tol):
        raise ValueError(
            "Stage-2 final reward alpha weights must sum to 1.0: "
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


def _compute_stage2_stability_terms(mc_eval, stability_proxy_std_ref, stability_weight, num_metrics):
    stability_proxy = 0.0
    stability_penalty = 0.0
    if mc_eval is None:
        return stability_proxy, stability_penalty

    std_components = [
        float(mc_eval.get("loss_std", 0.0)) / max(float(stability_proxy_std_ref), 1e-8),
        float(mc_eval.get("metric1_std", 0.0)) / max(float(stability_proxy_std_ref), 1e-8),
    ]
    if num_metrics > 1:
        std_components.append(
            float(mc_eval.get("metric2_std", 0.0)) / max(float(stability_proxy_std_ref), 1e-8)
        )
    stability_proxy = float(np.mean(std_components)) if std_components else 0.0
    stability_penalty = -float(stability_weight) * max(0.0, stability_proxy - 1.0)
    return stability_proxy, stability_penalty


def _compute_stage2_single_shot_reward_components(
    *,
    loss,
    metric1,
    metric2,
    baseline_metrics,
    constraint_limits,
    cost_value,
    cost_lower_bound,
    cost_upper_bound,
    perf_weights,
    barrier_weights,
    num_metrics,
    safe_rate_target=NOISE_STAGE_TRAIN_SAFE_RATE_TARGET,
):
    baseline_loss, baseline_metric1, baseline_metric2 = baseline_metrics
    limits = constraint_limits

    _MARGIN_DENOM_FLOOR = 0.01
    margin_loss = (float(limits["loss"]) - float(loss)) / max(
        abs(float(limits["loss"]) - float(baseline_loss)), _MARGIN_DENOM_FLOOR
    )
    margin_m1 = (float(metric1) - float(limits["metric1"])) / max(
        abs(float(baseline_metric1) - float(limits["metric1"])), _MARGIN_DENOM_FLOOR
    )
    if num_metrics > 1:
        margin_m2 = (float(metric2) - float(limits["metric2"])) / max(
            abs(float(baseline_metric2) - float(limits["metric2"])), _MARGIN_DENOM_FLOOR
        )
    else:
        margin_m2 = 0.0

    viol_loss = max(0.0, -margin_loss)
    viol_m1 = max(0.0, -margin_m1)
    viol_m2 = max(0.0, -margin_m2) if num_metrics > 1 else 0.0
    violation_penalty = (
        float(barrier_weights["loss"]) * viol_loss
        + float(barrier_weights["metric1"]) * viol_m1
    )
    if num_metrics > 1:
        violation_penalty += float(barrier_weights.get("metric2", 0.0)) * viol_m2

    util_loss = max(0.0, margin_loss)
    util_m1 = max(0.0, margin_m1)
    util_m2 = max(0.0, margin_m2) if num_metrics > 1 else 0.0
    margin_utility = (
        float(perf_weights["loss"]) * util_loss
        + float(perf_weights["metric1"]) * util_m1
    )
    if num_metrics > 1:
        margin_utility += float(perf_weights.get("metric2", 0.0)) * util_m2

    cost_score = float(np.clip(
        (float(cost_upper_bound) - float(cost_value))
        / max(float(cost_upper_bound) - float(cost_lower_bound), 1e-8),
        0.0, 1.0,
    ))
    cost_saving = (float(cost_upper_bound) - float(cost_value)) / max(
        float(cost_upper_bound), 1e-8
    )

    constraints_ok = (margin_loss >= 0.0) and (margin_m1 >= 0.0)
    if num_metrics > 1:
        constraints_ok = constraints_ok and (margin_m2 >= 0.0)

    safe_rate = 1.0 if constraints_ok else 0.0
    safe_rate_gap = max(0.0, float(safe_rate_target) - safe_rate)

    raw_reward = (
        float(NOISE_STAGE_TAIL_MARGIN_WEIGHT) * margin_utility
        - float(NOISE_STAGE_TAIL_VIOLATION_WEIGHT) * violation_penalty
        - float(NOISE_STAGE_SAFE_RATE_GAP_WEIGHT) * safe_rate_gap
        + float(NOISE_STAGE_MEAN_PERF_WEIGHT) * margin_utility
        + float(NOISE_STAGE_COST_WEIGHT) * cost_score
    )

    reward_components = {
        "loss_limit": float(limits["loss"]),
        "metric1_limit": float(limits["metric1"]),
        "metric2_limit": float(limits.get("metric2", limits["metric1"])),
        "loss_ratio": float(margin_loss),
        "metric1_ratio": float(margin_m1),
        "metric2_ratio": float(margin_m2),
        "perf_score": float(margin_utility),
        "cost_score": float(cost_score),
        "barrier_penalty": float(violation_penalty),
        "stability_proxy": 0.0,
        "stability_penalty": 0.0,
        "constraints_ok": bool(constraints_ok),
        "cost_lower_bound": float(cost_lower_bound),
        "cost_upper_bound": float(cost_upper_bound),
        "current_cost": float(cost_value),
        "cost_saving": float(cost_saving),
        "metric2_active": bool(num_metrics > 1),
        "safe_rate": float(safe_rate),
        "safe_rate_gap": float(safe_rate_gap),
        "tail_k": 1,
        "tail_violation_cvar": float(violation_penalty),
        "tail_margin_score": float(margin_utility),
        "mean_perf_score": float(margin_utility),
        "unsafe_count": 0 if constraints_ok else 1,
        "tail_loss_mean": float(loss),
        "tail_acc_mean": float(metric1),
        "tail_f1_mean": float(metric2) if num_metrics > 1 else float(metric1),
        "tail_margin_weight": float(NOISE_STAGE_TAIL_MARGIN_WEIGHT),
        "tail_violation_weight": float(NOISE_STAGE_TAIL_VIOLATION_WEIGHT),
        "safe_rate_gap_weight": float(NOISE_STAGE_SAFE_RATE_GAP_WEIGHT),
        "mean_perf_weight_coef": float(NOISE_STAGE_MEAN_PERF_WEIGHT),
        "cost_weight_coef": float(NOISE_STAGE_COST_WEIGHT),
        "safe_rate_target": float(safe_rate_target),
        "raw_tail_reward": float(raw_reward),
        "raw_final_reward": float(raw_reward),
        "final_selection_score": float(raw_reward),
        "terminal_reward_mode": "single_shot_margin",
    }
    return float(raw_reward), reward_components


def _compute_stage2_reward_components(
    *,
    loss,
    metric1,
    metric2,
    baseline_metrics,
    constraint_limits,
    cost_value,
    cost_lower_bound,
    cost_upper_bound,
    final_reward_alpha_perf,
    final_reward_alpha_cost,
    perf_weights,
    barrier_weights,
    stability_proxy_std_ref,
    stability_weight,
    num_metrics,
    mc_eval=None,
    tail_margin_weight=NOISE_STAGE_TAIL_MARGIN_WEIGHT,
    tail_violation_weight=NOISE_STAGE_TAIL_VIOLATION_WEIGHT,
    safe_rate_gap_weight=NOISE_STAGE_SAFE_RATE_GAP_WEIGHT,
    mean_perf_weight=NOISE_STAGE_MEAN_PERF_WEIGHT,
    cost_weight=NOISE_STAGE_COST_WEIGHT,
    safe_rate_target=NOISE_STAGE_TRAIN_SAFE_RATE_TARGET,
):
    """训练期 tail surrogate reward 计算器。

    基于 trial 级数组计算 tail violation/margin/safe_rate，
    组合为 R_tail-train；不再使用 stability_penalty。
    """
    baseline_loss, baseline_metric1, baseline_metric2 = baseline_metrics
    loss_limit = float(constraint_limits["loss"])
    metric1_limit = float(constraint_limits["metric1"])
    metric2_limit = float(constraint_limits.get("metric2", metric1_limit)) if num_metrics > 1 else metric1_limit

    # 均值 ratio 用于兼容字段
    # 使用 abs() 防止 limit 在 baseline 同侧时分母极小导致 ratio 爆炸
    _MARGIN_DENOM_FLOOR = 0.01
    loss_ratio = (loss_limit - float(loss)) / max(abs(loss_limit - float(baseline_loss)), _MARGIN_DENOM_FLOOR)
    metric1_ratio = (float(metric1) - metric1_limit) / max(abs(float(baseline_metric1) - metric1_limit), _MARGIN_DENOM_FLOOR)
    if num_metrics > 1:
        metric2_ratio = (float(metric2) - metric2_limit) / max(abs(float(baseline_metric2) - metric2_limit), _MARGIN_DENOM_FLOOR)
    else:
        metric2_ratio = 0.0

    constraints_ok = (
        (loss_ratio >= 0.0)
        and (metric1_ratio >= 0.0)
        and (num_metrics == 1 or metric2_ratio >= 0.0)
    )

    cost_score = float(
        np.clip(
            (float(cost_upper_bound) - float(cost_value))
            / max(float(cost_upper_bound) - float(cost_lower_bound), 1e-8),
            0.0,
            1.0,
        )
    )
    cost_saving = (float(cost_upper_bound) - float(cost_value)) / max(float(cost_upper_bound), 1e-8)

    # 如果有 trial 级数据，使用 tail surrogate；否则回退到均值方式
    trials = mc_eval.get("trials") if mc_eval else None
    if trials and len(trials) > 0:
        tail_info = _compute_tail_metrics_from_trials(
            trials, constraint_limits, baseline_metrics,
            perf_weights, barrier_weights, num_metrics,
        )
        safe_rate = tail_info["safe_rate"]
        tail_violation_cvar = tail_info["tail_violation_cvar"]
        tail_margin_score = tail_info["tail_margin_score"]
        mean_perf_score = tail_info["mean_perf_score"]
        tail_k = tail_info["tail_k"]
        unsafe_count = tail_info["unsafe_count"]
        tail_loss_mean = tail_info["tail_loss_mean"]
        tail_acc_mean = tail_info["tail_acc_mean"]
        tail_f1_mean = tail_info["tail_f1_mean"]
    else:
        # 无 trial 数据时的回退（兼容旧调用）
        loss_score = float(np.clip(loss_ratio, 0.0, 1.0))
        m1_score = float(np.clip(metric1_ratio, 0.0, 1.0))
        m2_score = float(np.clip(metric2_ratio, 0.0, 1.0)) if num_metrics > 1 else 0.0
        mean_perf_score = (
            float(perf_weights["loss"]) * loss_score
            + float(perf_weights["metric1"]) * m1_score
        )
        if num_metrics > 1:
            mean_perf_score += float(perf_weights.get("metric2", 0.0)) * m2_score
        safe_rate = 1.0 if constraints_ok else 0.0
        tail_violation_cvar = 0.0
        tail_margin_score = mean_perf_score
        tail_k = 0
        unsafe_count = 0 if constraints_ok else 1
        tail_loss_mean = float(loss)
        tail_acc_mean = float(metric1)
        tail_f1_mean = float(metric2) if num_metrics > 1 else float(metric1)

    # R_tail-train（文档 3.4）
    safe_rate_gap = max(0.0, float(safe_rate_target) - safe_rate)
    raw_tail_reward = (
        float(tail_margin_weight) * tail_margin_score
        - float(tail_violation_weight) * tail_violation_cvar
        - float(safe_rate_gap_weight) * safe_rate_gap
        + float(mean_perf_weight) * mean_perf_score
        + float(cost_weight) * cost_score
    )

    # 兼容旧字段
    perf_score = mean_perf_score
    barrier_penalty = tail_violation_cvar
    stability_proxy = 0.0
    stability_penalty = 0.0

    reward_components = {
        "loss_limit": float(loss_limit),
        "metric1_limit": float(metric1_limit),
        "metric2_limit": float(metric2_limit),
        "loss_ratio": float(loss_ratio),
        "metric1_ratio": float(metric1_ratio),
        "metric2_ratio": float(metric2_ratio),
        "perf_score": float(perf_score),
        "cost_score": float(cost_score),
        "barrier_penalty": float(barrier_penalty),
        "stability_proxy": float(stability_proxy),
        "stability_penalty": float(stability_penalty),
        "constraints_ok": bool(constraints_ok),
        "cost_lower_bound": float(cost_lower_bound),
        "cost_upper_bound": float(cost_upper_bound),
        "current_cost": float(cost_value),
        "cost_saving": float(cost_saving),
        "metric2_active": bool(num_metrics > 1),
        # 尾部安全核心指标
        "safe_rate": float(safe_rate),
        "safe_rate_gap": float(safe_rate_gap),
        "tail_k": int(tail_k),
        "tail_violation_cvar": float(tail_violation_cvar),
        "tail_margin_score": float(tail_margin_score),
        "mean_perf_score": float(mean_perf_score),
        "unsafe_count": int(unsafe_count),
        "tail_loss_mean": float(tail_loss_mean),
        "tail_acc_mean": float(tail_acc_mean),
        "tail_f1_mean": float(tail_f1_mean),
        # reward 权重
        "tail_margin_weight": float(tail_margin_weight),
        "tail_violation_weight": float(tail_violation_weight),
        "safe_rate_gap_weight": float(safe_rate_gap_weight),
        "mean_perf_weight_coef": float(mean_perf_weight),
        "cost_weight_coef": float(cost_weight),
        "safe_rate_target": float(safe_rate_target),
        "raw_tail_reward": float(raw_tail_reward),
        "raw_final_reward": float(raw_tail_reward),
        "final_selection_score": float(raw_tail_reward),
    }
    return float(raw_tail_reward), reward_components


def _build_candidate_score_sort_key(
    final_selection_score,
    stability_score,
    cost_value,
    loss_value,
    metric_sum,
):
    return (
        -float(final_selection_score),
        float(stability_score),
        float(cost_value),
        float(loss_value),
        -float(metric_sum),
    )


def _write_noise_step_info(step_info, f):
    f.write(f"  全局步数（step_global）: {step_info['step_global']}\n")
    f.write(f"  回合编号（episode_id）: {step_info['episode_id']}\n")
    f.write(f"  层索引（layer_index）: {step_info['layer_index']}\n")
    f.write(f"  状态向量（state_vector）: {step_info['state_vector']}\n")
    f.write(f"  当前输入噪声缩放因子（curr_input_noise_scaling_factor）: {step_info['curr_input_noise_scaling_factor']}\n")
    f.write(f"  当前wq噪声缩放因子（curr_wq_noise_scaling_factor）: {step_info['curr_wq_noise_scaling_factor']}\n")
    f.write(f"  当前wk噪声缩放因子（curr_wk_noise_scaling_factor）: {step_info['curr_wk_noise_scaling_factor']}\n")
    f.write(f"  当前wv噪声缩放因子（curr_wv_noise_scaling_factor）: {step_info['curr_wv_noise_scaling_factor']}\n")
    f.write(f"  当前wo噪声缩放因子（curr_wo_noise_scaling_factor）: {step_info['curr_wo_noise_scaling_factor']}\n")
    f.write(f"  当前wffn1噪声缩放因子（curr_wffn1_noise_scaling_factor）: {step_info['curr_wffn1_noise_scaling_factor']}\n")
    f.write(f"  当前wffn2噪声缩放因子（curr_wffn2_noise_scaling_factor）: {step_info['curr_wffn2_noise_scaling_factor']}\n")
    f.write(f"  输入x概率分布（x_prob_dist）: {step_info['x_prob_dist']}\n")
    f.write(f"  wq概率分布（wq_prob_dist）: {step_info['wq_prob_dist']}\n")
    f.write(f"  wk概率分布（wk_prob_dist）: {step_info['wk_prob_dist']}\n")
    f.write(f"  wv概率分布（wv_prob_dist）: {step_info['wv_prob_dist']}\n")
    f.write(f"  wo概率分布（wo_prob_dist）: {step_info['wo_prob_dist']}\n")
    f.write(f"  wffn1概率分布（wffn1_prob_dist）: {step_info['wffn1_prob_dist']}\n")
    f.write(f"  wffn2概率分布（wffn2_prob_dist）: {step_info['wffn2_prob_dist']}\n")
    f.write(f"  评论家值（critic_value）: {step_info['critic_value']}\n")
    f.write(f"  累计成本（accumulated_cost）: {step_info['accumulated_cost']}\n")
    f.write(f"  输入噪声配置（input_noise_config）: {step_info['input_noise_config']}\n")
    f.write(f"  wq噪声配置（wq_noise_config）: {step_info['wq_noise_config']}\n")
    f.write(f"  wk噪声配置（wk_noise_config）: {step_info['wk_noise_config']}\n")
    f.write(f"  wv噪声配置（wv_noise_config）: {step_info['wv_noise_config']}\n")
    f.write(f"  wo噪声配置（wo_noise_config）: {step_info['wo_noise_config']}\n")
    f.write(f"  wffn1噪声配置（wffn1_noise_config）: {step_info['wffn1_noise_config']}\n")
    f.write(f"  wffn2噪声配置（wffn2_noise_config）: {step_info['wffn2_noise_config']}\n")
    if "current_lr" in step_info:
        f.write(f"  当前学习率（current_lr）: {step_info['current_lr']:.6f}\n")
    if "current_entropy_coef" in step_info:
        f.write(f"  当前熵系数（current_entropy_coef）: {step_info['current_entropy_coef']:.6f}\n")
    if step_info.get("mc_samples") is not None:
        f.write(f"  蒙特卡洛采样数（mc_samples）: {step_info['mc_samples']}\n")
        f.write(f"  蒙特卡洛损失均值（mc_loss_mean）: {step_info['mc_loss_mean']}\n")
        f.write(f"  蒙特卡洛损失标准差（mc_loss_std）: {step_info['mc_loss_std']}\n")
        f.write(f"  蒙特卡洛指标1均值（mc_metric1_mean）: {step_info['mc_metric1_mean']}\n")
        f.write(f"  蒙特卡洛指标1标准差（mc_metric1_std）: {step_info['mc_metric1_std']}\n")
        f.write(f"  蒙特卡洛指标2均值（mc_metric2_mean）: {step_info['mc_metric2_mean']}\n")
        f.write(f"  蒙特卡洛指标2标准差（mc_metric2_std）: {step_info['mc_metric2_std']}\n")
    if step_info.get("step_reward") is not None:
        f.write(f"  步奖励（step_reward）: {step_info['step_reward']}\n")
    if step_info.get("dense_reward_step") is not None:
        f.write(f"  稠密奖励步（dense_reward_step）: {step_info['dense_reward_step']}\n")
    if step_info.get("raw_final_reward") is not None:
        f.write(f"  原始最终奖励（raw_final_reward）: {step_info['raw_final_reward']}\n")
    if step_info.get("final_selection_score") is not None:
        f.write(f"  最终选择分数（final_selection_score）: {step_info['final_selection_score']}\n")
    if step_info.get("accumulated_dense_reward") is not None:
        f.write(f"  累计稠密奖励（accumulated_dense_reward）: {step_info['accumulated_dense_reward']}\n")
    if step_info.get("stability_proxy") is not None:
        f.write(f"  稳定性代理（stability_proxy）: {step_info['stability_proxy']}\n")
    if step_info.get("stability_penalty") is not None:
        f.write(f"  稳定性惩罚（stability_penalty）: {step_info['stability_penalty']}\n")
    # dense probe 详情（文档修改方案 第8步）
    if step_info.get("partial_probe_loss") is not None:
        f.write(f"  部分探测损失（partial_probe_loss）: {step_info['partial_probe_loss']}\n")
    if step_info.get("partial_probe_metric1") is not None:
        f.write(f"  部分探测指标1（partial_probe_metric1）: {step_info['partial_probe_metric1']}\n")
    if step_info.get("partial_probe_metric2") is not None:
        f.write(f"  部分探测指标2（partial_probe_metric2）: {step_info['partial_probe_metric2']}\n")
    if step_info.get("partial_margin_loss") is not None:
        f.write(f"  部分余量损失（partial_margin_loss）: {step_info['partial_margin_loss']}\n")
    if step_info.get("partial_margin_metric1") is not None:
        f.write(f"  部分余量指标1（partial_margin_metric1）: {step_info['partial_margin_metric1']}\n")
    if step_info.get("partial_margin_metric2") is not None:
        f.write(f"  部分余量指标2（partial_margin_metric2）: {step_info['partial_margin_metric2']}\n")
    if step_info.get("partial_completion_score") is not None:
        f.write(f"  部分补全分数（partial_completion_score）: {step_info['partial_completion_score']}\n")
    if step_info.get("partial_violation_penalty") is not None:
        f.write(f"  部分违约惩罚（partial_violation_penalty）: {step_info['partial_violation_penalty']}\n")
    if step_info.get("remaining_budget_ratio") is not None:
        f.write(f"  剩余预算比例（remaining_budget_ratio）: {step_info['remaining_budget_ratio']}\n")
    if step_info.get("completion_policy") is not None:
        f.write(f"  补全策略（completion_policy）: {step_info['completion_policy']}\n")


def _write_warning_report(warning_file, warnings, stage_label=""):
    """将奖励骤降警告写入报告；每次调用覆盖式重写全部已有警告。"""
    import datetime
    dt_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    title = "强化学习奖励骤降警告报告"
    meta_lines = [
        f"阶段:     {stage_label}",
        f"更新时间: {dt_str}",
        f"累计警告: {len(warnings)} 条",
    ]
    inner_w = max(len(title), max((len(s) for s in meta_lines), default=0), 60)
    header_bar = "═" * (inner_w + 4)
    with open(warning_file, "w", encoding="utf-8") as f:
        f.write(f"╔{header_bar}╗\n")
        f.write(f"║  {title.center(inner_w)}║\n")
        f.write(f"╠{header_bar}╣\n")
        for line in meta_lines:
            f.write(f"║  {line.ljust(inner_w)}║\n")
        f.write(f"╚{header_bar}╝\n\n")

        for i, w in enumerate(warnings, 1):
            ep_start, ep_end = w["episode_range"]
            drop_pct = (
                abs(w["drop"] / w["prev_avg"]) * 100
                if abs(w["prev_avg"]) > 1e-8 else float("inf")
            )
            block_lines = [
                f"⚠ 警告 #{i}  ·  {w['type']}",
                f"",
                f"  检测窗口:          第 {w['window']} 个 PPO 更新窗口",
                f"  涉及回合:          第 {ep_start} 轮 ~ 第 {ep_end} 轮",
                f"  上次窗口平均奖励:  {w['prev_avg']:.4f}",
                f"  本次窗口平均奖励:  {w['curr_avg']:.4f}",
                f"  下降幅度:          {w['drop']:.4f}  ({drop_pct:.1f}%)  （告警阈值: {w['threshold']:.2f}）",
                f"",
                f"  请查看以下详情文件定位具体回合信息:",
            ]
            for df in w["detail_files"]:
                block_lines.append(f"    → details/{df}")
            block_lines.extend([
                f"",
                f"  排查建议:",
                f"    1. 打开上述详情文件，搜索 raw_final_reward 查找异常回合",
                f"    2. 检查该回合的约束余量（margin）是否出现极端值",
                f"    3. 检查成本（cost）是否急剧变化",
                f"    4. 对比前后窗口的动作概率分布是否发生突变",
            ])
            _write_rounded_box_to_file(f, block_lines)
            f.write("\n")

        f.write("═" * (inner_w + 4) + "\n")
        f.write(f"报告更新于 {dt_str}  ·  共 {len(warnings)} 条警告\n")


def _ppo_update_noise_gtrxl(evaluator, noise_net, optimizer, buffer, device,
                            mini_batch_episodes=8, entropy_coef=None,
                            ppo_update_step=0,
                            ppo_eps_clip=0.2, ppo_k_epochs=4,
                            ppo_value_coef=0.5,
                            gtrxl_warmup_mode="constant",
                            gtrxl_warmup_updates=0,
                            gtrxl_short_warmup_updates=20,
                            gtrxl_entropy_lower_bound=0.005,
                            gtrxl_mini_batch_episodes=8,
                            value_clip_range=0.2,
                            mean_aux_loss_coef=NOISE_STAGE_MEAN_AUX_LOSS_COEF,
                            mean_perf_normalizer=None):
    """双头 critic PPO 更新（文档修改方案 第7步）。

    rewards = 新的 dense delta reward + terminal single-shot reward。
    mean_perf_targets = 每步 partial probe score（不再是终点 mean_perf）。
    actor 仍然学最大化累积回报；主 critic 仍然学回报值函数；
    辅助 critic 不再学"最后才冒出来的 mean_perf"，而学"当前前缀 completion quality"。
    """
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
     old_logprobs, rewards, values, dones, mean_perf_targets) = buffer.get_batch(device)

    n_eps = cont_features.size(0)
    all_advantages = []
    all_returns = []
    for i in range(n_eps):
        adv, ret = evaluator.compute_gae(
            rewards[i].cpu().numpy(),
            values[i].cpu().numpy(),
            dones[i].cpu().numpy(),
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
        dtype=torch.float32
    ).to(device)
    values_normalized = torch.tensor(
        evaluator.return_normalizer.normalize(values.cpu().numpy()),
        dtype=torch.float32
    ).to(device)

    # 辅助 critic target 归一化（与主 critic 解耦，避免梯度尺度不匹配）
    if mean_perf_normalizer is not None:
        mean_perf_normalizer.update(mean_perf_targets)
        mean_perf_targets_normalized = torch.tensor(
            mean_perf_normalizer.normalize(mean_perf_targets.cpu().numpy()),
            dtype=torch.float32
        ).to(device)
    else:
        mean_perf_targets_normalized = mean_perf_targets

    last_policy_loss = 0.0
    last_value_loss = 0.0
    last_tail_value_loss = 0.0
    last_mean_aux_loss = 0.0
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
            mb_mean_perf = mean_perf_targets_normalized[mb_idx]

            new_logprobs, entropy, new_tail_values, new_mean_aux_values = noise_net.evaluate_actions(
                mb_cont, mb_layer, mb_prev_actions, mb_actions
            )

            new_logprobs_flat = new_logprobs.reshape(-1)
            entropy_flat = entropy.reshape(-1)
            new_tail_flat = new_tail_values.reshape(-1)
            new_mean_aux_flat = new_mean_aux_values.reshape(-1)
            mb_old_lp_flat = mb_old_lp.reshape(-1)
            mb_adv_flat = mb_adv.reshape(-1)
            mb_ret_flat = mb_ret.reshape(-1)
            mb_old_val_flat = mb_old_val.reshape(-1)
            mb_mean_perf_flat = mb_mean_perf.reshape(-1)

            # 策略损失
            ratios = torch.exp(new_logprobs_flat - mb_old_lp_flat)
            surr1 = ratios * mb_adv_flat
            surr2 = torch.clamp(ratios, 1 - ppo_eps_clip, 1 + ppo_eps_clip) * mb_adv_flat
            policy_loss = -torch.min(surr1, surr2).mean()

            # Tail value head 损失（主 critic，clipped + Huber）
            new_tail_norm = (new_tail_flat - evaluator.return_normalizer.mean) / (evaluator.return_normalizer.std + 1e-8)
            value_clipped = mb_old_val_flat + torch.clamp(
                new_tail_norm - mb_old_val_flat,
                -value_clip_range, value_clip_range
            )
            huber_loss_fn = nn.HuberLoss(reduction="none", delta=1.0)
            vl_unclipped = huber_loss_fn(new_tail_norm, mb_ret_flat)
            vl_clipped = huber_loss_fn(value_clipped, mb_ret_flat)
            tail_value_loss = torch.max(vl_unclipped, vl_clipped).mean()

            # Mean aux head 损失（辅助监督，Huber，归一化空间）
            if mean_perf_normalizer is not None:
                new_mean_aux_norm = (new_mean_aux_flat - mean_perf_normalizer.mean) / (mean_perf_normalizer.std + 1e-8)
            else:
                new_mean_aux_norm = new_mean_aux_flat
            mean_aux_loss = huber_loss_fn(new_mean_aux_norm, mb_mean_perf_flat).mean()

            total_value_loss = tail_value_loss + float(mean_aux_loss_coef) * mean_aux_loss

            mean_entropy = entropy_flat.mean()
            effective_entropy_coef = entropy_coef
            if mean_entropy.item() < gtrxl_entropy_lower_bound:
                entropy_deficit = gtrxl_entropy_lower_bound - mean_entropy.item()
                effective_entropy_coef = entropy_coef + 10.0 * entropy_deficit

            entropy_loss = -mean_entropy
            loss = policy_loss + ppo_value_coef * total_value_loss + effective_entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(noise_net.parameters(), 0.5)
            optimizer.step()

            last_policy_loss = policy_loss.item()
            last_value_loss = total_value_loss.item()
            last_tail_value_loss = tail_value_loss.item()
            last_mean_aux_loss = mean_aux_loss.item()
            last_entropy = mean_entropy.item()

    return last_policy_loss, last_value_loss, last_entropy


def _plot_noise_training_curves(
        evaluator,
        episode_returns, episode_raw_final_rewards, episode_losses, episode_metric1s, episode_metric2s,
        episode_entropies,
        base_loss, base_p, base_s,
        training_curve_path="noise_ppo_training_curve.png",
        entropy_curve_path="noise_ppo_entropy_curve.png",
        ppo_update_interval=170,
        use_validation=True):
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

        dataset_info = f" ({evaluator.data_path})"
        val_guided_info = " [Validation Guided]" if use_validation else ""

        if _num_m == 1:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle(f"Noise PPO Training Curves{dataset_info}{val_guided_info}", fontsize=14, fontweight="bold")
            ax1, ax2, ax3 = axes
            ax4 = None
        else:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f"Noise PPO Training Curves{dataset_info}{val_guided_info}", fontsize=14, fontweight="bold")
            ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

        ax1.plot(episodes, episode_returns_arr, label="Episode Return", alpha=0.45, color="steelblue")
        ax1.plot(episodes_ma, episode_returns_ma, label=f"Episode Return MA ({window})", linewidth=2, color="navy")
        ax1.plot(episodes, raw_final_rewards, label="Raw Final Reward", alpha=0.45, color="darkorange")
        ax1.plot(episodes_ma, raw_final_rewards_ma, label=f"Raw Final Reward MA ({window})", linewidth=2, color="orangered")
        ax1.set_xlabel("Episode"); ax1.set_ylabel("Reward"); ax1.set_title("Episode Return vs Raw Final Reward"); ax1.grid(True, alpha=0.3); ax1.legend()

        ax2.plot(episodes, losses, label="Loss", alpha=0.6, color="red")
        ax2.plot(episodes_ma, losses_ma, label=f"Moving Avg ({window})", linewidth=2, color="darkred")
        ax2.set_xlabel("Episode"); ax2.set_ylabel("Loss"); ax2.set_title("Loss (lower is better)"); ax2.grid(True, alpha=0.3)
        ax2.axhline(y=base_loss, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Baseline")
        ax2.legend()

        ax3.plot(episodes, metric1s, label=_m1_name, alpha=0.6, color="green")
        ax3.plot(episodes_ma, metric1s_ma, label=f"Moving Avg ({window})", linewidth=2, color="darkgreen")
        ax3.set_xlabel("Episode"); ax3.set_ylabel(_m1_name); ax3.set_title(f"{_m1_name} (higher is better)"); ax3.grid(True, alpha=0.3)
        ax3.axhline(y=base_p, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Baseline")
        ax3.legend()

        if ax4 is not None:
            ax4.plot(episodes, metric2s, label=_m2_name, alpha=0.6, color="purple")
            ax4.plot(episodes_ma, metric2s_ma, label=f"Moving Avg ({window})", linewidth=2, color="darkviolet")
            ax4.set_xlabel("Episode"); ax4.set_ylabel(_m2_name); ax4.set_title(f"{_m2_name} (higher is better)"); ax4.grid(True, alpha=0.3)
            ax4.axhline(y=base_s, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Baseline")
            ax4.legend()

        plt.tight_layout()
        training_dir = os.path.dirname(training_curve_path)
        if training_dir:
            os.makedirs(training_dir, exist_ok=True)
        plt.savefig(training_curve_path, dpi=150)
        plt.close()
        evaluator.log(f"  [绘图] 噪声 PPO 训练曲线已写入 path={training_curve_path}")

        if episode_entropies:
            update_episodes = np.arange(ppo_update_interval, len(episode_returns) + 1, ppo_update_interval)
            entropies = np.array(episode_entropies, dtype=np.float32)
            if len(update_episodes) == len(entropies):
                fig_ent, ax_ent = plt.subplots(1, 1, figsize=(10, 5))
                ax_ent.plot(update_episodes, entropies, label="Policy Entropy", alpha=0.8, color="teal", marker="o", markersize=3)
                window_ent = min(5, max(1, len(entropies) // 5))
                if len(entropies) >= window_ent:
                    kernel_ent = np.ones(window_ent, dtype=np.float32) / window_ent
                    ent_ma = np.convolve(entropies, kernel_ent, mode="valid")
                    ax_ent.plot(update_episodes[window_ent - 1:], ent_ma, label=f"Moving Avg ({window_ent})", linewidth=2, color="darkgreen")
                ax_ent.set_xlabel("Episode (at PPO update)")
                ax_ent.set_ylabel("Entropy")
                ax_ent.set_title("Noise PPO Training: Policy Entropy over Episodes")
                ax_ent.grid(True, alpha=0.3)
                ax_ent.legend()
                plt.tight_layout()
                entropy_dir = os.path.dirname(entropy_curve_path)
                if entropy_dir:
                    os.makedirs(entropy_dir, exist_ok=True)
                plt.savefig(entropy_curve_path, dpi=150)
                plt.close()
                evaluator.log(f"  [绘图] 噪声 PPO 熵曲线已写入 path={entropy_curve_path}")
    except Exception as e:
        evaluator.log(f"  [警告] 绘制噪声 PPO 训练曲线失败（plot failed）: {e}")


def _plot_noise_risk_curves(
        evaluator,
        episode_safe_rates, episode_tail_violation_cvars, episode_tail_margin_scores,
        confirm_window_indices, confirm_search_safe_rates, confirm_holdout_safe_rates,
        confirm_search_tail_cvars, confirm_holdout_tail_cvars,
        risk_curve_path="noise_risk_curves.png",
        confirm_curve_path="noise_confirm_curves.png"):
    """绘制尾部风险曲线图和候选确认曲线图（文档 9.3）。"""
    if len(episode_safe_rates) == 0:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        episodes = np.arange(1, len(episode_safe_rates) + 1)
        window = min(50, max(1, len(episode_safe_rates) // 10))

        def _ma(data):
            arr = np.array(data, dtype=np.float32)
            if len(arr) < window:
                return arr
            kernel = np.ones(window, dtype=np.float32) / window
            return np.convolve(arr, kernel, mode="valid")

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("噪声阶段风险曲线（Noise Stage Risk Curves）", fontsize=14, fontweight="bold")

        sr = np.array(episode_safe_rates, dtype=np.float32)
        axes[0].plot(episodes, sr, alpha=0.4, color="green")
        axes[0].plot(episodes[window-1:] if len(sr) >= window else episodes, _ma(sr), linewidth=2, color="darkgreen")
        axes[0].axhline(y=NOISE_STAGE_TRAIN_SAFE_RATE_TARGET, color="red", linestyle="--", alpha=0.7, label=f"目标（target）={NOISE_STAGE_TRAIN_SAFE_RATE_TARGET}")
        axes[0].set_xlabel("回合（Episode）"); axes[0].set_ylabel("安全率（safe_rate）"); axes[0].set_title("训练期安全率（Train Safe Rate）")
        axes[0].grid(True, alpha=0.3); axes[0].legend()

        vc = np.array(episode_tail_violation_cvars, dtype=np.float32)
        axes[1].plot(episodes, vc, alpha=0.4, color="red")
        axes[1].plot(episodes[window-1:] if len(vc) >= window else episodes, _ma(vc), linewidth=2, color="darkred")
        axes[1].set_xlabel("回合（Episode）"); axes[1].set_ylabel("尾部违约CVaR"); axes[1].set_title("训练期尾部违约CVaR")
        axes[1].grid(True, alpha=0.3)

        tm = np.array(episode_tail_margin_scores, dtype=np.float32)
        axes[2].plot(episodes, tm, alpha=0.4, color="blue")
        axes[2].plot(episodes[window-1:] if len(tm) >= window else episodes, _ma(tm), linewidth=2, color="navy")
        axes[2].set_xlabel("回合（Episode）"); axes[2].set_ylabel("尾部余量（tail_margin）"); axes[2].set_title("训练期尾部余量分数")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        risk_dir = os.path.dirname(risk_curve_path)
        if risk_dir:
            os.makedirs(risk_dir, exist_ok=True)
        plt.savefig(risk_curve_path, dpi=150)
        plt.close()
        evaluator.log(f"  [绘图] 噪声风险曲线已写入 path={risk_curve_path}")

        if confirm_window_indices:
            fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
            fig2.suptitle("候选确认曲线（Confirm Curves）", fontsize=14, fontweight="bold")
            wi = np.array(confirm_window_indices)
            axes2[0].plot(wi, confirm_search_safe_rates, marker="o", markersize=3, label="搜索集（search）", color="green")
            if confirm_holdout_safe_rates:
                axes2[0].plot(wi, confirm_holdout_safe_rates, marker="s", markersize=3, label="留出集（holdout）", color="blue")
            axes2[0].axhline(y=NOISE_STAGE_BEST_TEST_SAFE_RATE_MIN, color="red", linestyle="--", alpha=0.7)
            axes2[0].set_xlabel("窗口（Window）"); axes2[0].set_ylabel("安全率（safe_rate）"); axes2[0].set_title("确认阶段安全率")
            axes2[0].grid(True, alpha=0.3); axes2[0].legend()

            axes2[1].plot(wi, confirm_search_tail_cvars, marker="o", markersize=3, label="搜索集（search）", color="red")
            if confirm_holdout_tail_cvars:
                axes2[1].plot(wi, confirm_holdout_tail_cvars, marker="s", markersize=3, label="留出集（holdout）", color="orange")
            axes2[1].axhline(y=NOISE_STAGE_BEST_TEST_CVAR_MAX, color="gray", linestyle="--", alpha=0.7)
            axes2[1].set_xlabel("窗口（Window）"); axes2[1].set_ylabel("尾部违约CVaR"); axes2[1].set_title("确认阶段尾部违约CVaR")
            axes2[1].grid(True, alpha=0.3); axes2[1].legend()

            plt.tight_layout()
            confirm_dir = os.path.dirname(confirm_curve_path)
            if confirm_dir:
                os.makedirs(confirm_dir, exist_ok=True)
            plt.savefig(confirm_curve_path, dpi=150)
            plt.close()
            evaluator.log(f"  [绘图] 候选确认曲线已写入 path={confirm_curve_path}")
    except Exception as e:
        evaluator.log(f"  [警告] 绘制风险/确认曲线失败（plot failed）: {e}")
