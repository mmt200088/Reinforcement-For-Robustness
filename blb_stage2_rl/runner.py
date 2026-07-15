"""BLBStage2RLRunner —— 顶层入口，对接 ``LayerImportanceEvaluator``。

接口与 ``noise_rl_module_v2.NoiseRLModuleV2`` 保持一致：
    runner = BLBStage2RLRunner(evaluator)
    result_dict = runner.run(fixed_gelu, fixed_softmax, fixed_label, fixed_source,
                             resume_checkpoint_path=...)

返回的 ``result_dict`` 与旧版兼容，使下游 ``UnifiedFinalEvaluationModule`` 等
消费保持不变。

设计要点：
  * 训练前先调用 ``evaluator.apply_configuration(fixed_gelu, fixed_softmax)``
    把多项式近似装好，再调用 BLB；这样 spec §3.2 的 attn / GELU 替换前置依赖
    自动满足。
  * 训练结束后调用 ``bridge.clear()``（env.step 内已经做）+ 显式
    ``handler.restore_layer_block*_noise`` 防御式还原；旧版 final_eval 仍可
    在干净的多项式近似模型上运行。
  * 新版 RL 强依赖真实 ``Rescale_optimizer`` 子项目；CKKS 模数链、fusion 与
    total_bits 必须由该算法计算，初始化失败会直接中止训练。
"""
from __future__ import annotations

import json
import math
import os
import pickle
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from cli_parse_utils import parse_int_list_text
from rescale_optimizer_bridge import RescaleOptimizerBridge
from .action_mask import (
    action_allowed,
    action_mask_hash,
    build_action_mask,
    build_baseline_action_bias,
    ensure_action_allowed,
    load_action_mask_file,
)
from .action_space import (
    K_LEVELS,
    MaxSFsTable,
    action_dims_for_config,
    action_vector_to_cfgs,
    avg_truncation_k_in_action,
    describe_action_vector,
    layer_dims,
)
from .env import (
    BLBStage2Env,
    BLBStage2EnvConfig,
    ProbeBatch,
    estimate_baseline_cost_stats,
)
from .policy import (
    BLBStage2Policy,
    PPOConfig,
    RolloutBuffer,
    ppo_update,
)
from .reward import (
    BaselineCostStats,
    RewardWeights,
    calibrate_weights_from_baseline,
    compute_reward,
)
from .persistence import (
    BLBRewardCrashWatcher,
    BLBStatusBoard,
    BLBStepDetailsWriter,
    append_blb_episode_trace_row,
    dump_crash_report,
    write_action_description_files,
    write_blb_final_report,
    write_training_curves,
)
from .baseline_bootstrap import (
    load_static_skeletons_baseline,
    static_skeletons_baseline_to_action,
)


def _normalize_supported_rl_algo(value: Any, *, context: str = "rl_algo") -> str:
    algo = str(value or "ppo").strip().lower()
    if algo != "ppo":
        raise ValueError(
            "GRPO has been disabled for this project after the PPO-vs-GRPO "
            f"MRPC generalization study. {context} must be 'ppo', got {value!r}."
        )
    return "ppo"


BLB_STAGE2_LIVE_CHECKPOINT_FILENAME = "blb_stage2_rl_checkpoint_live.pt"
BLB_STAGE2_FINAL_CHECKPOINT_FILENAME = "blb_stage2_rl_checkpoint_final.pt"
BLB_STAGE2_BEST_CFG_FILENAME = "blb_stage2_best_cfg.pkl"

# BLB Stage 2 progress belongs inside the active run output directory.
# Fallbacks also stay under Parting Chapter/persistent so this path replaces
# the old rl_results/persistent layout instead of creating a side directory.
BLB_PARTING_CHAPTER_DIRNAME = "Parting Chapter"
BLB_PERSISTENT_DIRNAME = "persistent"


def _resolve_repo_root() -> str:
    """回到项目根目录。``runner.py`` 位于 ``<root>/blb_stage2_rl/``，向上一级即可。"""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(here)


def resolve_blb_persistence_dir(evaluator) -> str:
    """计算 BLB Stage 2 RL 的持久化目录（覆盖 ``ev.noise_stage_progress_dir``）。

    输出形如 ``<run_output_dir>/stage2_noise/progress``。
    若 ``evaluator.run_output_dir`` 为空，使用 ``Parting Chapter/persistent/blb_stage2_default_run``。
    """
    run_dir = str(getattr(evaluator, "run_output_dir", "") or "").strip()
    if run_dir:
        if getattr(evaluator, "decoupled_layout", False):
            # 解耦扁平布局：stage2 工作目录直接放 progress/（无 stage2_noise/ 嵌套）。
            out = os.path.join(run_dir, "progress")
        else:
            out = os.path.join(run_dir, "stage2_noise", "progress")
    else:
        repo_root = _resolve_repo_root()
        out = os.path.join(
            repo_root,
            BLB_PARTING_CHAPTER_DIRNAME,
            BLB_PERSISTENT_DIRNAME,
            "blb_stage2_default_run",
            "stage2_noise",
            "progress",
        )
    os.makedirs(out, exist_ok=True)
    return out


def _effective_probe_batch_count(ev, train_cfg) -> int:
    """Return enough mini-batches to cover stage2_probe_size unless overridden."""
    explicit = getattr(ev, "blb_v3_probe_batch_count", None)
    if explicit not in (None, ""):
        acc_threshold_resolution: Optional[BaselineMetricThreshold] = None
        baseline_preflight_metrics: Dict[str, Any] = {}
        try:
            return max(1, int(explicit))
        except Exception:
            pass
    probe_size = max(1, int(getattr(ev, "stage2_probe_size", 256)))
    batch_size = max(1, int(getattr(ev, "batch_size", 1)))
    return max(1, int(math.ceil(float(probe_size) / float(batch_size))))


def _selection_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


@dataclass(frozen=True)
class BaselineMetricThreshold:
    threshold: float
    source: str
    allowed_drop: float
    raw_baseline_metric: float
    all_max_blb_metric: float


def _baseline_derived_metric_threshold(
        *,
        current_threshold: Any,
        raw_baseline_metric: Any,
        all_max_blb_metric: Any,
        allowed_drop: Any,
        ) -> BaselineMetricThreshold:
    """Resolve a metric limit from the all-max BLB baseline unless explicit.

    ``allowed_drop`` is a relative tolerance: 0.001 means a 0.1% metric drop.
    """
    current = _selection_float(current_threshold, 0.0)
    raw_metric = _selection_float(raw_baseline_metric, 0.0)
    blb_metric = _selection_float(all_max_blb_metric, raw_metric)
    drop = max(0.0, _selection_float(allowed_drop, 0.0))
    if math.isfinite(current) and current > 0.0:
        return BaselineMetricThreshold(
            threshold=float(current),
            source="explicit",
            allowed_drop=float(drop),
            raw_baseline_metric=float(raw_metric),
            all_max_blb_metric=float(blb_metric),
        )
    return BaselineMetricThreshold(
        threshold=max(0.0, float(blb_metric) * (1.0 - float(drop))),
        source="baseline_derived_all_max_blb",
        allowed_drop=float(drop),
        raw_baseline_metric=float(raw_metric),
        all_max_blb_metric=float(blb_metric),
    )


def _candidate_selection_key(
        reward: float,
        breakdown: Optional[Mapping[str, Any]],
        ) -> Tuple[float, float, float, float]:
    if not isinstance(breakdown, Mapping):
        return (0.0, 0.0, 0.0, _selection_float(reward, -float("inf")))
    invalid_rank = 0.0 if bool(breakdown.get("invalid", False)) else 1.0
    acc_violation = max(0.0, _selection_float(breakdown.get("acc_violation", 0.0), 0.0))
    stab_violation = max(0.0, _selection_float(breakdown.get("stab_violation", 0.0), 0.0))
    return (
        -acc_violation,
        -stab_violation,
        invalid_rank,
        _selection_float(reward, -float("inf")),
    )


def is_better_blb_candidate(
        *,
        candidate_reward: float,
        candidate_breakdown: Optional[Mapping[str, Any]],
        best_reward: float,
        best_breakdown: Optional[Mapping[str, Any]],
        ) -> bool:
    """Compare BLB candidates by accuracy, stability, then optimizer cost."""
    return _candidate_selection_key(candidate_reward, candidate_breakdown) > _candidate_selection_key(
        best_reward,
        best_breakdown,
    )


def _baseline_preflight_stability_threshold(
        *,
        current_threshold: float,
        observed_loss_std: Any,
        tolerance: Any,
        ) -> float:
    """Keep the stability gate at least as wide as the noisy all-max baseline."""
    current = _selection_float(current_threshold, float("inf"))
    observed = _selection_float(observed_loss_std, 0.0)
    tol = max(0.0, _selection_float(tolerance, 0.0))
    if observed <= 0.0:
        return current
    floor = observed * (1.0 + tol) + 1e-12
    if not math.isfinite(current):
        return floor
    return max(current, floor)


def _allowed_neighbor_indices(
        *,
        kind: str,
        baseline_idx: int,
        dim: int,
        radius: int,
        ) -> List[int]:
    """Allowed local moves around the all-max baseline for one action slot."""
    baseline_idx = int(baseline_idx)
    dim = int(dim)
    radius = max(0, int(radius))
    if dim <= 0:
        return []
    if str(kind) == "K":
        base_k = int(K_LEVELS[baseline_idx])
        candidates = [
            idx for idx, value in enumerate(K_LEVELS)
            if int(value) <= base_k
        ]
        candidates.sort(key=lambda idx: int(K_LEVELS[idx]), reverse=True)
        return [int(idx) for idx in candidates[:radius + 1]]
    lo = max(0, baseline_idx - radius)
    hi = min(dim - 1, baseline_idx)
    return list(range(lo, hi + 1))


def _neighborhood_curriculum(
        *,
        episode_offset: int,
        ramp_episodes: int,
        max_mutations: int,
        max_radius: int,
        ) -> Tuple[int, int]:
    ramp = max(1, int(ramp_episodes))
    progress = min(1.0, max(0.0, float(episode_offset) / float(ramp)))
    mutations = 1 + int(math.floor(progress * max(0, int(max_mutations) - 1)))
    radius = 1 + int(math.floor(progress * max(0, int(max_radius) - 1)))
    return max(1, mutations), max(1, radius)


def _build_kind_drop_action(
        baseline_action_vec: np.ndarray,
        baseline_records: Sequence[Mapping[str, Any]],
        action_dim_by_index: Sequence[int],
        *,
        kinds: Sequence[str],
        radius: int = 1,
        ) -> Tuple[np.ndarray, List[int]]:
    """Build a coordinated one-step lower action for the selected slot kinds."""
    action = np.asarray(baseline_action_vec, dtype=int).copy()
    target_kinds = {str(k) for k in kinds}
    touched: List[int] = []
    for idx, record in enumerate(baseline_records):
        if idx >= len(action_dim_by_index) or idx >= action.size:
            continue
        kind = str(record.get("kind", ""))
        if kind not in target_kinds:
            continue
        if not bool(record.get("effective", True)):
            continue
        dim = int(action_dim_by_index[idx])
        if dim <= 1:
            continue
        allowed = _allowed_neighbor_indices(
            kind=kind,
            baseline_idx=int(action[idx]),
            dim=dim,
            radius=int(radius),
        )
        non_baseline = [
            int(value) for value in allowed
            if 0 <= int(value) < dim and int(value) != int(action[idx])
        ]
        if not non_baseline:
            continue
        action[idx] = int(non_baseline[0])
        touched.append(int(idx))
    return action, touched


def _warmstart_action_mode(
        *,
        episode_index: int,
        anchor_episodes: int,
        cost_probe_count: int,
        neighbor_sampling: bool,
        has_mutable_neighbors: bool,
        neighbor_ramp_episodes: int,
        ) -> Tuple[str, int]:
    """Choose warmstart action source from absolute training progress."""
    episode_offset = max(0, int(episode_index))
    anchor_episodes = max(0, int(anchor_episodes))
    cost_probe_count = max(0, int(cost_probe_count))
    neighbor_ramp_episodes = max(0, int(neighbor_ramp_episodes))
    if episode_offset < anchor_episodes:
        return "anchor", -1
    cost_probe_index = episode_offset - anchor_episodes
    if 0 <= cost_probe_index < cost_probe_count:
        return "cost_probe", int(cost_probe_index)
    if (
            bool(neighbor_sampling)
            and bool(has_mutable_neighbors)
            and episode_offset < neighbor_ramp_episodes
    ):
        return "neighbor", -1
    return "policy", -1


def _effective_warmstart_anchor_episodes(
        *,
        configured_anchor_episodes: Optional[int],
        rollout_size: int,
        total_episodes: int,
        start_episode: int,
        disable_warmstart_on_resume: bool = False,
        ) -> int:
    if bool(disable_warmstart_on_resume) and int(start_episode) > 0:
        return 0
    if configured_anchor_episodes is None:
        anchor = int(rollout_size)
    else:
        anchor = int(configured_anchor_episodes)
    return max(0, min(anchor, int(total_episodes)))


_BLOCK_ERROR_RE = re.compile(
    r"(?P<label>[A-Za-z0-9_]+_L\d+):\s*(?P<detail>.*?)(?=(?:;\s*)?[A-Za-z0-9_]+_L\d+:\s*|$)"
)


def _translate_rescale_error_detail(detail: str) -> str:
    text = str(detail or "").strip().strip(";").strip()
    omitted_suffix = ""
    omitted_tail = re.search(r";\s*(\.\.\. \(\+[0-9]+ more\))$", text)
    if omitted_tail:
        omitted_suffix = "；" + _translate_rescale_error_detail(omitted_tail.group(1))
        text = text[:omitted_tail.start()].strip()
    qmax_match = re.fullmatch(
        r"new chain has prime\(s\) > q_max=([0-9]+) at stage\(s\) (\[[^\]]+\]); "
        r"fusion cannot reduce\. Reject\.?",
        text,
    )
    if qmax_match:
        return (
            f"新模数链出现大于 q_max={qmax_match.group(1)} 的素数（prime）；"
            f"位置在阶段（stage）{qmax_match.group(2)}；融合（fusion）无法降低，已拒绝。"
            f"{omitted_suffix}"
        )
    qmin_match = re.fullmatch(
        r"replan FAILED: a prime < q_min=([0-9]+) could not be fused after "
        r"([0-9]+) successful fusion\(s\)\.?",
        text,
    )
    if qmin_match:
        return (
            f"重新规划（replan）失败：存在小于 q_min={qmin_match.group(1)} 的素数（prime）；"
            f"已完成 {qmin_match.group(2)} 次成功融合（fusion），仍无法继续融合。"
            f"{omitted_suffix}"
        )
    omitted_match = re.fullmatch(r"\.\.\. \(\+([0-9]+) more\)", text)
    if omitted_match:
        return f"另有 {omitted_match.group(1)} 项被上游摘要省略。"
    return text


def _format_blb_episode_error_log(episode: int, error_text: str) -> str:
    raw = str(error_text or "").strip()
    prefix = "Rescale_optimizer invalid blocks:"
    body = raw
    if body.startswith(prefix):
        body = body[len(prefix):].strip()
    entries = [
        (match.group("label").strip(), _translate_rescale_error_detail(match.group("detail")))
        for match in _BLOCK_ERROR_RE.finditer(body)
    ]
    lines = [
        "  【BLB 单回合错误】",
        f"    - 回合（episode）：{int(episode)}",
        "    - 来源：Rescale_optimizer 约束校验",
    ]
    if entries:
        lines.append(f"    - 失败位置：共 {len(entries)} 个 block")
        for idx, (label, detail) in enumerate(entries, start=1):
            lines.append(f"      {idx}. {label}：{detail}")
    else:
        lines.append(f"    - 原始信息：{_translate_rescale_error_detail(raw)}")
    return "\n".join(lines)


def _format_warmstart_cost_probe_log(warmstart_cost_probe_actions) -> str:
    lines = ["  * 预热（warmstart）成本探针："]
    for name, _action, touched in warmstart_cost_probe_actions:
        lines.append(f"    - {name}：影响槽位 {len(touched)} 个")
    return "\n".join(lines)


def _format_blb_rollout_summary_log(
        *,
        update_count: int,
        episode: int,
        total_episodes: int,
        reward_mean: float,
        reward_max: float,
        reward_min: float,
        invalid_count: int,
        priority_counts: Mapping[int, int],
        anchor_count: int,
        cost_probe_count: int,
        neighborhood_count: int,
        policy_loss: float,
        value_loss: float,
        entropy: float,
        clip_fraction: float,
        entropy_by_kind: Optional[Mapping[str, float]] = None,
        raw_entropy_by_kind: Optional[Mapping[str, float]] = None,
        masked_entropy_by_kind: Optional[Mapping[str, float]] = None,
        ) -> str:
    def _entropy_text(values: Optional[Mapping[str, float]]) -> str:
        items = []
        for key, value in (values or {}).items():
            try:
                items.append(f"{key}={float(value):.2f}")
            except Exception:
                items.append(f"{key}={value}")
        return ", ".join(items) if items else "none"

    entropy_items = []
    for key, value in (entropy_by_kind or {}).items():
        try:
            entropy_items.append(f"{key}={float(value):.2f}")
        except Exception:
            entropy_items.append(f"{key}={value}")
    entropy_text = "，".join(entropy_items) if entropy_items else "暂无"
    return "\n".join([
        "  【BLB Rollout 汇总】",
        f"    - PPO 更新轮次（update）：{int(update_count)}",
        f"    - 训练进度（episode）：{int(episode)} / {int(total_episodes)}",
        f"    - 奖励（reward，均值 / 最大 / 最小）：{float(reward_mean):.3f} / {float(reward_max):.3f} / {float(reward_min):.3f}",
        (
            "    - 优先级计数（P0/P1/P2/P3）："
            f"P0 无效={int(invalid_count)}，"
            f"P1 精度未达标={int(priority_counts.get(1, 0))}，"
            f"P2 稳定性未达标={int(priority_counts.get(2, 0))}，"
            f"P3 约束通过={int(priority_counts.get(3, 0))}"
        ),
        (
            "    - 动作来源（A/C/N）："
            f"A 基线锚点={int(anchor_count)}，"
            f"C 成本探针={int(cost_probe_count)}，"
            f"N 邻域采样={int(neighborhood_count)}"
        ),
        (
            "    - PPO 指标："
            f"policy_loss={float(policy_loss):.4f}，"
            f"value_loss={float(value_loss):.4f}，"
            f"entropy={float(entropy):.4f}，"
            f"clip_fraction={float(clip_fraction):.4f}"
        ),
        f"    - 槽位熵（H_kind）：{entropy_text}",
        f"    - raw_entropy_by_kind: {_entropy_text(raw_entropy_by_kind)}",
        f"    - masked_entropy_by_kind: {_entropy_text(masked_entropy_by_kind)}",
    ])


def _format_blb_best_log(
        *,
        episode: int,
        best_reward: float,
        previous_reward_label: Optional[str],
        priority: Optional[int],
        diff_text: str = "",
        ) -> str:
    lines = [
        "  【BLB 新最佳】",
        f"    - 回合（episode）：{int(episode)}",
        f"    - 当前奖励（reward）：{float(best_reward):.4f}",
    ]
    if previous_reward_label:
        lines.append(f"    - 上一最佳奖励：{previous_reward_label}")
    else:
        lines.append("    - 上一最佳奖励：无，这是第一个候选最佳。")
    if priority is not None:
        lines.append(f"    - 优先级（priority）：P{int(priority)}")
    if diff_text:
        diff_text = re.sub(
            r"\.\.\. \(\+([0-9]+) more\)",
            r"另有 \1 个变化未展开",
            str(diff_text),
        )
        lines.append(f"    - 变化位置：{diff_text}")
    return "\n".join(lines)


def _format_blb_train_iter_log(
        *,
        episode: int,
        total_episodes: int,
        return_mean: float,
        return_max: float,
        best_reward: float,
        policy_loss: float,
        value_loss: float,
        entropy: float,
        clip_fraction: float,
        ) -> str:
    return "\n".join([
        "  【BLB 训练迭代】",
        f"    - 训练进度（episode）：{int(episode)} / {int(total_episodes)}",
        f"    - 近期回报（return，均值 / 最大）：{float(return_mean):+.3f} / {float(return_max):+.3f}",
        f"    - 历史最佳：best_reward={float(best_reward):.3f}",
        (
            "    - PPO 指标："
            f"policy_loss={float(policy_loss):.4f}，"
            f"value_loss={float(value_loss):.4f}，"
            f"entropy={float(entropy):.4f}，"
            f"clip_fraction={float(clip_fraction):.4f}"
        ),
    ])


# ---------------------------------------------------------------------------
# CLI 友好的 RL 训练超参（给 evaluator / rl_tune 透传）
# ---------------------------------------------------------------------------
@dataclass
class BLBStage2TrainConfig:
    """``BLBStage2RLRunner`` 的可配置训练参数。"""
    total_episodes: int = 0                 # 0 = train until natural convergence
    rollout_size: int = 120                 # 每多少 episode 触发一次 PPO update
    seed: int = 42
    eval_interval: int = 100                # 多少 episode 跑一次 deterministic eval
    save_interval: int = 200
    profile: str = "default"
    # spec §6.4 / §3.1
    acc_threshold: float = 0.0              # baseline 精度往下浮 1pp 后用此值
    stab_threshold: float = float("inf")
    # PPO
    ppo: PPOConfig = field(default_factory=PPOConfig)
    # PPO-only. Legacy fields remain so old checkpoints/presets deserialize,
    # but any non-PPO value is rejected at construction and runner handoff.
    rl_algo: str = "ppo"
    grpo_kl_beta: float = 0.0

    def __post_init__(self) -> None:
        self.rl_algo = _normalize_supported_rl_algo(
            self.rl_algo, context="BLBStage2TrainConfig.rl_algo"
        )
        self.grpo_kl_beta = 0.0
        self.validate_decision_granularity()
        self.validate_reward_design()
        self.validate_robust_constraint_config()

    def validate_decision_granularity(self) -> str:
        from .layerwise_runner import normalize_decision_granularity

        value = normalize_decision_granularity(self.decision_granularity)
        self.decision_granularity = value
        return value

    def validate_reward_design(self) -> str:
        from .layerwise_runner import normalize_reward_design

        value = normalize_reward_design(self.reward_design)
        self.reward_design = value
        return value

    def validate_robust_constraint_config(self) -> None:
        for field_name in (
                "baseline_groups", "baseline_trials_per_group",
                "constraint_bootstrap_samples", "promotion_validation_trials",
                "final_selection_validation_trials",
        ):
            value = int(getattr(self, field_name))
            if value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
            setattr(self, field_name, value)
        self.stage2_stability_multiplier = float(self.stage2_stability_multiplier)
        if self.stage2_stability_multiplier <= 0.0:
            raise ValueError("stage2_stability_multiplier must be positive")
        for field_name in (
                "online_constraint_probability", "promotion_constraint_probability",
                "final_constraint_probability",
        ):
            value = float(getattr(self, field_name))
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{field_name} must be in (0, 1]")
            setattr(self, field_name, value)
        if not (
                self.online_constraint_probability
                <= self.promotion_constraint_probability
                <= self.final_constraint_probability
        ):
            raise ValueError(
                "constraint probabilities must satisfy online <= promotion <= final"
            )
    # 环境
    # Bumped 3→5 on 2026-05-18: 3 trials gave loss_std a ~50% sampling error,
    # making one outlier trial blow up the std and trip priority-2 (stability)
    # falsely. 5 trials reduces the relative SE to ~35% so loss_std rank-orders
    # actions more reliably. ~+67% per-step forward compute, +30% total
    # wall-time. See diagnostics_summary.md (s1t0.005 run) for the symptom.
    num_trials_per_step: int = 5
    probe_batch_count: int = 4
    # 自动校准
    calibrate_baseline_samples: int = 8
    # Real Rescale_optimizer parameters. BLB Stage-2 RL intentionally has no
    # heuristic/stub/subprocess training path.
    inproc_rescale_optimizer_root: Optional[str] = field(
        default_factory=lambda: os.path.join(_resolve_repo_root(), "Rescale_optimizer")
    )
    inproc_profile: Optional[str] = None                 # e.g. "mrpc"；用于自动定位 configs/<profile>
    inproc_configs: Optional[Mapping[str, str]] = None   # {config_name: graph_json_path}；不传则按 profile 自动扫
    inproc_baseline_archive: Optional[str] = None        # 不传则 <root>/configs/<profile>/static_skeletons_<profile>.json
    warmstart_baseline_bias: bool = False
    warmstart_bias_gain: float = 1.2
    # Per-slot baseline prior is now external and decays per episode in the
    # sequential GTrXL path. 1.2 is only the fresh/anchor prior; it falls to a
    # weak 0.15 safety prior after episode 2000 so policy learning and
    # empirical Pareto proposals can leave the baseline neighborhood.
    warmstart_anchor_episodes: Optional[int] = 0
    warmstart_neighbor_sampling: bool = False
    warmstart_neighbor_ramp_episodes: Optional[int] = None
    warmstart_neighbor_max_mutations: int = 12
    warmstart_neighbor_max_radius: int = 1
    guarded_radius2_enabled: bool = False
    guarded_radius2_min_episode: int = 1060
    guarded_radius2_stall_window: int = 600
    guarded_radius2_health_window: int = 100
    guarded_radius2_max_mutations: int = 4
    guarded_radius2_episode_fraction: float = 0.15
    guarded_radius2_cooldown_episodes: int = 300
    guarded_radius2_min_radius1_successes: int = 3
    static_invalid_level_mask_enabled: bool = False
    disable_warmstart_on_resume: bool = False
    action_mask_enabled: bool = False
    action_mask_mode: str = "none"
    action_mask_file: Optional[str] = None
    action_mask_baseline_logit_bonus: float = 0.0
    action_mask_source: str = ""
    # ---- per-block sequential RL knobs (additive; default ON 2026-05-15) ----
    sequential_rl: bool = True
    """If True, the runner replaces the legacy single-shot rollout loop with the
    horizon-N per-block sequential loop (BLBStage2SequentialEnv +
    BLBStage2SequentialPolicy + train_sequential). Default flipped to True on
    2026-05-15 -- the single-shot path now requires explicit opt-out via
    ``--blb-v3-sequential-rl false`` in the launcher."""
    sequential_invalid_penalty: float = 1.0
    sequential_cost_shaping_coeff: float = 0.0
    sequential_fusion_shaping_coeff: float = 0.0
    sequential_early_terminate_on_invalid: bool = False
    # 2026-05-18 (sampling-collapse hotfix): the PPO entropy bonus was
    # actively undoing the forced-baseline anchor. Schedule ent_coef to be
    # 0 during anchor episodes, then linearly ramp to ``ppo.ent_coef`` over
    # ``ent_coef_ramp_episodes`` sample episodes. See SequentialTrainConfig
    # docstring in blb_stage2_rl/sequential_runner.py for full rationale.
    ent_coef_anchor: float = 0.0
    ent_coef_ramp_episodes: int = 600
    # ADR-015/Stage-1 alignment: Stage-1-style reward plus Stage-2 stability,
    # with Stage-1's high→low cosine entropy schedule.
    reward_design: str = "robust_constrained"
    ent_coef_schedule: str = "cosine"
    ent_coef_cosine_start: float = 0.05
    ent_coef_cosine_end: float = 0.001
    ent_coef_cosine_plateau: float = 0.25
    ent_coef_cosine_lower_bound: float = 0.012
    # force_baseline_episodes: 0 -> use auto-default 60 inside
    # run_sequential_via_runner. Surfaced here so a preset can pin a
    # specific anchor length without relying on the auto-default.
    force_baseline_episodes: int = 0
    # 2026-05-19: two-GPU reward-probe parallelism. Empty / single-element list
    # → single-GPU codepath unchanged. Two or more device ids → BLBStage2Env
    # gets a ProbeRunner that fans the K trials across these devices in
    # parallel. Element 0 must be the primary device (where the env's existing
    # model lives). Wired through --blb-v3-reward-devices "0,1" in the launcher.
    reward_devices: List[int] = field(default_factory=list)
    # Episode-parallel rollout devices (--stage2-rl-devices, fusion mode only,
    # 2026-06-10): N workers each run complete episodes (rollout + replan +
    # serial K-trial probe) on their own model replica with global-episode
    # seeding. Mutually exclusive with reward_devices. Empty → legacy loop.
    stage2_rl_devices: List[int] = field(default_factory=list)
    # Workers per device for episode-parallel rollout (2026-06-12): >1 overlaps
    # one worker's CPU rollout/bookkeeping with a sibling's GPU-bound probe on
    # the same card. Results stay byte-identical for any value (per-device RNG
    # atomic-unit locks; episode results depend only on the global index).
    stage2_workers_per_device: int = 1
    # Fast online reward mode: collect terminal actions and evaluate distinct
    # actions concurrently across reward_devices. Default online K mirrors the
    # normal K=5 training reward; CLI can still override it explicitly.
    fast_reward_mode_enabled: bool = False
    online_num_trials_per_step: int = 5
    terminal_eval_batch_size: int = 4
    promotion_validation_trials: int = 25
    promotion_margin_window: float = 0.25
    final_selection_top_n: int = 20
    final_selection_validation_trials: int = 25
    baseline_groups: int = 5
    baseline_trials_per_group: int = 5
    constraint_bootstrap_samples: int = 4096
    online_constraint_probability: float = 0.50
    promotion_constraint_probability: float = 0.80
    final_constraint_probability: float = 0.95
    stage2_stability_multiplier: float = 2.0
    # ---- 4-sub-stage mode (opt-in 2026-05-27) -----------------------------
    # When ``substage_mode`` is True, ``BLBStage2RLRunner.run`` dispatches to
    # ``substage_runner.run_substage_via_runner`` instead of the legacy
    # per-block sequential path. Each sub-stage trains one block in
    # ``substage_block_order``; ``substage_frozen_blocks`` lists blocks that
    # are pinned to the ``static_skeletons`` baseline throughout (block 3 by
    # design). See ``substage_runner.py`` for the budget allocator
    # (progressive re-baseline with hard floor at ``acc_orig - tol``).
    substage_mode: bool = False
    substage_block_order: List[int] = field(default_factory=lambda: [1, 2, 4, 5])
    substage_frozen_blocks: List[int] = field(default_factory=lambda: [3])
    substage_episodes_each: int = 15000
    substage_promotion_top_k: int = 5
    substage_promotion_trials: int = 8
    # ---- Fusion-count action (opt-in 2026-06-03) --------------------------
    # When True, the sequential RL path decides per block (fusion_option, K) via
    # the offline blb_stage2_rl/fusion_maps/<profile>/ map instead of all per-slot
    # SF heads. Disables safe-neighbor / guarded-radius2 / invalid masks (the map
    # holds only valid configs). Mutually exclusive with substage_mode.
    fusion_count_action: bool = True
    decision_granularity: str = "layer"
    # Fusion-mode block-granularity safe-neighbor curriculum (additive 2026-06-05).
    # Default ON: gently widens how many blocks may leave the baseline (option 0,
    # baseline K) each episode, dissolving to the unrestricted open mask after the
    # ramp (so the full action space stays reachable). Set False for the A/B
    # control group (unrestricted from the start, the pre-2026-06-05 behaviour).
    fusion_neighbor_curriculum_enabled: bool = False
    fusion_neighbor_ramp_episodes: int = 0
    fusion_neighbor_max_radius: int = 6
    # Scheduled forced-fusion probes (ADR-011 2026-06-11): every N post-anchor
    # episodes one episode forces fusion option 1 on one rotating block type
    # (block2 -> block5 -> block4) at baseline K, keeping fresh on-policy
    # fusion evidence flowing after the curriculum dissolves. 0 disables.
    fusion_probe_interval: int = 0
    # ADR-012 exploration floor for the fusion option / K slots (0 disables).
    fusion_exploration_epsilon: float = 0.0
    fusion_exploration_epsilon_k: float = 0.0
    # ---- COINN-style OSR pre-prune (opt-in 2026-05-27) ---------------------
    # Empty osr_results_path → no OSR layer (legacy behaviour). When set, the
    # runner loads existing results from PATH, or runs a fresh scan saving to
    # PATH if absent. When ``osr_scan_only`` is True, training exits after the
    # scan; otherwise the scan results are applied as an extra mask in the
    # PPO retry loop, alongside the existing three (Static / Empirical /
    # Forbidden) masks. See blb_stage2_rl/osr.py for details.
    osr_results_path: str = ""
    osr_scan_only: bool = False
    osr_num_combo_samples: int = 300
    osr_allow_fingerprint_mismatch: bool = False


# ---------------------------------------------------------------------------
# Result key 与旧版兼容：legacy *_scaling_factors 全 max baseline + BLB 详细字段
# ---------------------------------------------------------------------------
def _build_legacy_compatible_best_noise_config(evaluator) -> Dict[str, np.ndarray]:
    """构造与旧版 ``*_scaling_factors`` 字段同形的 baseline-shape dict。

    新版 BLB RL 的最优策略是 BLB 噪声配置，与 legacy ``*_scaling_factors`` 不同。
    为了让下游 ``UnifiedFinalEvaluationModule`` 不崩，这里返回全 max baseline ──
    final-eval 会以 baseline 评估（等同 stage2 没找到更优配置）。BLB 真正的
    最优配置另外保存在 ``best_blb_action_vector`` / ``best_blb_cfgs`` / 落盘
    ``best_policy/blb_stage2_best_cfg.pkl`` 中。
    """
    return evaluator._get_max_noise_configuration()


# ---------------------------------------------------------------------------
# 主 Runner
# ---------------------------------------------------------------------------
class BLBStage2RLRunner:
    """加强版 BLB Stage 2 强化学习训练入口。"""

    def __init__(self, evaluator):
        self.evaluator = evaluator

    # ------------------------------------------------------------------
    # 主接口（与 noise_rl_module_v2.NoiseRLModuleV2.run 兼容）
    # ------------------------------------------------------------------
    def run(
            self,
            fixed_gelu,
            fixed_softmax,
            fixed_label,
            fixed_source,
            resume_checkpoint_path=None,
            ) -> Dict[str, Any]:
        ev = self.evaluator
        # 用 ASCII bullet 而非 ▸（U+25B8）以兼容 Windows GBK 控制台 ── evaluator
        # 内部仍以 UTF-8 写日志文件，所以这里换成 "*" 不影响日志文件内容。
        bullet = "*"
        log = self._make_log_safe(ev.log)

        # ---------- 0) 解析配置 ----------
        train_cfg = self._build_train_config_from_evaluator(ev)
        # 2026-05-15：per-block sequential RL is the default Stage-2 path.
        # Dispatch to the sequential runner before the heavy single-shot setup
        # so the two paths stay genuinely independent. The single-shot loop
        # below is reachable only when ``train_cfg.sequential_rl`` is False.
        # 2026-05-27: 4-sub-stage path takes priority over the per-block
        # sequential one when explicitly enabled. Both are sequential at heart;
        # substage_mode further restricts each "round" to one block (block 3
        # frozen) so GTrXL focuses on layer-to-layer relations.
        if bool(getattr(train_cfg, "substage_mode", False)):
            from .substage_runner import run_substage_via_runner
            return run_substage_via_runner(
                runner=self,
                train_cfg=train_cfg,
                fixed_gelu=fixed_gelu,
                fixed_softmax=fixed_softmax,
                fixed_label=fixed_label,
                fixed_source=fixed_source,
                resume_checkpoint_path=resume_checkpoint_path,
            )
        if bool(getattr(train_cfg, "sequential_rl", False)):
            from .sequential_runner import run_sequential_via_runner
            return run_sequential_via_runner(
                runner=self,
                train_cfg=train_cfg,
                fixed_gelu=fixed_gelu,
                fixed_softmax=fixed_softmax,
                fixed_label=fixed_label,
                fixed_source=fixed_source,
                resume_checkpoint_path=resume_checkpoint_path,
            )
        # ---------- 0.1) 切换到 BLB Stage 2 RL 持久化目录 ----------
        # BLB 进度文件写入当前 run_output_dir/stage2_noise/progress。
        legacy_progress_dir = str(getattr(ev, "noise_stage_progress_dir", "") or "")
        blb_progress_dir = resolve_blb_persistence_dir(ev)
        try:
            ev.noise_stage_progress_dir = blb_progress_dir
        except Exception:
            pass
        fixed_label_display = str(fixed_label)
        if fixed_label_display == "Stage-1 config (json)":
            fixed_label_display = "一阶段配置（Stage-1 config, json）"

        log("\n" + "=" * 80)
        log("【阶段 5：BLB Stage-2 噪声强化学习（v3）】")
        log("=" * 80)
        log(f"  {bullet} 固定 GELU/Softmax 来源：{fixed_source}；标签：{fixed_label_display}")
        log(f"  {bullet} GELU 离散阶数向量:   {np.asarray(fixed_gelu, dtype=int).tolist()}")
        log(f"  {bullet} Softmax 离散阶数向量: {np.asarray(fixed_softmax, dtype=int).tolist()}")
        log(f"  {bullet} 训练概览：数据集配置（profile）= {train_cfg.profile!r}    "
            f"总回合数（episode）= {train_cfg.total_episodes}    "
            f"PPO 更新间隔（rollout_size）= {train_cfg.rollout_size}")
        log(f"  {bullet} BLB 持久化目录：{blb_progress_dir}")
        if legacy_progress_dir and os.path.normpath(legacy_progress_dir) != os.path.normpath(blb_progress_dir):
            log(f"  {bullet} （旧 stage 2 RL 持久化目录 {legacy_progress_dir} 已停止使用，仅保留为历史归档）")

        # ---------- 0.2) 初始化训练状态板（支持 live tail） ----------
        run_basename = os.path.basename(os.path.normpath(str(getattr(ev, "run_output_dir", "") or ""))) or "blb_stage2_default_run"
        status = BLBStatusBoard(
            blb_progress_dir,
            total_episodes=int(train_cfg.total_episodes),
            profile=str(train_cfg.profile),
            run_basename=run_basename,
            extra_meta={
                "fixed_label": str(fixed_label),
                "fixed_source": str(fixed_source),
                "rescale_optimizer": "in_process_real",
                "rescale_optimizer_root": str(train_cfg.inproc_rescale_optimizer_root),
            },
            log_fn=log,
        )
        status.set_phase("装载 stage1 GELU/Softmax 多项式近似")
        log(f"  {bullet} 状态板 JSON = {status.path}（训练期间持续刷新，可 live tail）")

        # Stage2 root (parent of progress/) — host of details/ + warning.txt
        # so the layout matches legacy noise_rl_module_v2 outputs.
        blb_stage2_root = os.path.dirname(os.path.normpath(blb_progress_dir))
        details_writer = BLBStepDetailsWriter(
            blb_stage2_root,
            batch_size=max(int(train_cfg.rollout_size) * 3, 360),
            log_fn=log,
        )
        crash_watcher = BLBRewardCrashWatcher(
            blb_stage2_root,
            drop_threshold=0.3,
            log_fn=log,
        )
        log(
            f"  {bullet} 详细诊断：{os.path.join(blb_stage2_root, 'details')}/ "
            f"（每 {details_writer._batch_size} 回合一文件，记录每回合错误/动作变化）"
        )
        log(
            f"  {bullet} 奖励暴跌警告：{os.path.join(blb_stage2_root, 'warning.txt')} "
            f"（PPO rollout 平均奖励较上一次跌幅 > {crash_watcher._drop_threshold:.2f} 时记录）"
        )

        if os.environ.get("BLB_NOISE_INSTALL_LOGS") is None:
            os.environ["BLB_NOISE_INSTALL_LOGS"] = "0"
            log("  * BLB 单候选安装日志：默认关闭；如需启用，请设置 BLB_NOISE_INSTALL_LOGS=1。")

        # ---------- 1) 应用 stage1 GELU/Softmax 多项式近似 ----------
        fixed_gelu = np.asarray(fixed_gelu, dtype=int)
        fixed_softmax = np.asarray(fixed_softmax, dtype=int)
        ev.apply_configuration(fixed_gelu, fixed_softmax)

        # 防御式：清掉所有 legacy 噪声残留 + 旧 BLB 残留
        try:
            ev.reversible_handler.restore_layer_input_noise(
                layer_indices=list(range(ev.total_layers)),
            )
        except Exception:
            pass

        # ---------- 2) 准备评估子集（probe） ----------
        probe_batches = self._build_probe_batches(ev, train_cfg)
        train_cfg.probe_batch_count = max(1, int(len(probe_batches) or train_cfg.probe_batch_count))
        probe_sample_count = sum(int(getattr(b.labels, "numel", lambda: 0)()) for b in probe_batches)
        log(
            f"  {bullet} 评估子集：batch 数 = {len(probe_batches)}；"
            f"样本数 = {probe_sample_count} / 请求值（requested）{int(getattr(ev, 'stage2_probe_size', 256))}"
        )

        # ---------- 3) 准备 RescaleOptimizer 桥 ----------
        rescale_bridge = self._build_rescale_bridge(train_cfg, log=log)

        # ---------- 4) 准备 max_sfs 表 + Env ----------
        # Baseline 只能来自 Rescale_optimizer 的 static_skeletons archive。
        # 文件缺失或 graph_key 不完整时直接让异常向上抛出，停止训练。
        ss_baseline_obj = load_static_skeletons_baseline(
            rescale_optimizer_root=str(train_cfg.inproc_rescale_optimizer_root),
            dataset=str(train_cfg.profile),
            num_layers=int(ev.total_layers),
            gelu_per_layer=[int(x) for x in np.asarray(fixed_gelu, dtype=int).reshape(-1)],
            softmax_per_layer=[int(x) for x in np.asarray(fixed_softmax, dtype=int).reshape(-1)],
        )
        _ss_action_vec, max_sfs, ss_cost_stats, ss_diag = static_skeletons_baseline_to_action(
            ss_baseline_obj,
            snap_sf_to_noise_table=False,
        )
        log(
            f"  {bullet} Baseline 来源（baseline source）：static_skeletons archive\n"
            f"      路径 = {ss_baseline_obj.archive_path}\n"
            f"      (block, layer) 数 = {ss_baseline_obj.aggregate_valid_block_count} (= 5*L - 1)\n"
            f"      active slots（含 fresh/encode/rescale）= {ss_diag['active_slot_count']}"
            f"  [fresh={ss_diag['fresh_slot_count']}, "
            f"encode={ss_diag['encode_slot_count']}, "
            f"rescale={ss_diag['rescale_slot_count']}]\n"
            f"      RO baseline 'off' rescale slots = {len(ss_diag['inactive_rescale_slots'])}"
        )
        if ss_diag["unmapped_nodes"]["propagation"]:
            log(
                f"      [info] {len(ss_diag['unmapped_nodes']['propagation'])} propagation_deltas "
                f"未映射回 RL 字段（通常是 CTCT_MUL 的 delta，例 ctct_rot_softmax_mul_v）"
            )
        if ss_diag["unmapped_nodes"]["rescale"]:
            log(
                f"      [info] {len(ss_diag['unmapped_nodes']['rescale'])} rescale 节点未映射"
                f"（n5/n6 graph 多出的 square 槽位被丢弃）"
            )
        per_layer_each_dim = layer_dims()
        env = BLBStage2Env(
            handler=ev.reversible_handler,
            model=ev.model,
            probe_batches=probe_batches,
            rescale_bridge=rescale_bridge,
            baseline=BaselineCostStats(),     # 占位，下面会覆盖
            reward_weights=RewardWeights(),   # 占位，下面会覆盖
            acc_threshold=train_cfg.acc_threshold,
            stab_threshold=train_cfg.stab_threshold,
            max_sfs=max_sfs,
            num_layers=int(ev.total_layers),
            gelu_degree=fixed_gelu,
            attn_degree=fixed_softmax,
            layers_attribute="model." + ev.layers_attribute,
            is_regression=bool(getattr(ev, "is_regression", False)),
            env_cfg=BLBStage2EnvConfig(
                profile=train_cfg.profile,
                num_trials_per_step=train_cfg.num_trials_per_step,
                probe_batch_count=train_cfg.probe_batch_count,
            ),
        )
        degree_sync = env.sync_degree_vectors_from_model()
        if degree_sync:
            log(f"  {bullet} Model degree sync: {degree_sync}")

        # ---------- 4.5) Multi-GPU reward-probe runner (opt-in) ----------
        if train_cfg.reward_devices and len(train_cfg.reward_devices) >= 2:
            from .probe_runner import build_probe_runner
            log(
                f"  {bullet} Multi-GPU reward probe enabled: "
                f"devices={train_cfg.reward_devices}"
            )
            env.probe_runner = build_probe_runner(
                primary_model=ev.model,
                primary_handler=ev.reversible_handler,
                primary_bridge=env.bridge,
                primary_probe_batches=env.probe_batches,
                layers_attribute="model." + ev.layers_attribute,
                is_regression=bool(getattr(ev, "is_regression", False)),
                device_ids=list(train_cfg.reward_devices),
                metric_profile=str(train_cfg.profile),
                log_fn=lambda m: log(f"  {bullet} {m}"),
            )

        # ---------- 5) baseline + reward 权重校准 ----------
        status.set_phase("校准 baseline cost / reward 权重")
        # 如 ss_cost_stats 存在（static_skeletons 路径成功），把它作为权威 baseline
        # 喂给 estimate_baseline_cost_stats —— RO 不再被调用第二次拿 baseline cost。
        # random-sample 用于估计 typical_*_drop 仍然走 in-process invoker。
        precomputed = {
            "total_bits_sum": int(ss_cost_stats.total_bits_sum),
            "total_fusion_count": int(ss_cost_stats.total_fusion_count),
            "avg_k": float(ss_cost_stats.avg_k),
        }
        baseline = estimate_baseline_cost_stats(
            env,
            sample_count=int(train_cfg.calibrate_baseline_samples),
            precomputed_baseline_signals=precomputed,
        )
        env.baseline = baseline
        baseline_status_entry = {
            "total_bits_sum": int(baseline.total_bits_sum),
            "total_fusion_count": int(baseline.total_fusion_count),
            "avg_k": float(baseline.avg_k),
            "typical_bits_drop": float(baseline.typical_bits_drop),
            "typical_fusion_count": float(baseline.typical_fusion_count),
            "typical_k_drop": float(baseline.typical_k_drop),
            "baseline_source": "static_skeletons_archive",
        }
        baseline_status_entry["static_skeletons_path"] = ss_baseline_obj.archive_path
        baseline_status_entry["active_slot_count"] = int(ss_diag["active_slot_count"])
        baseline_status_entry["inactive_rescale_slots_count"] = int(
            len(ss_diag["inactive_rescale_slots"])
        )
        status.set_baseline(baseline_status_entry)
        log(
            f"  {bullet} 基线成本（baseline cost）：total_bits_sum={baseline.total_bits_sum}, "
            f"total_fusion_count={baseline.total_fusion_count}, avg_k={baseline.avg_k:.2f}"
        )

        # 估计 baseline 精度 + 稳定性，用于硬阈值校准
        baseline_metrics = self._estimate_baseline_metrics(env)
        baseline.loss_mean = float(baseline_metrics.loss_mean)
        baseline.loss_std = float(baseline_metrics.loss_std)
        baseline.metric1_mean = float(baseline_metrics.metric1_mean)
        baseline.metric2_mean = float(baseline_metrics.metric2_mean)
        # v3 stability: combined_stab_excess needs baseline.metric{1,2}_std too.
        baseline.metric1_std = float(getattr(baseline_metrics, "metric1_std", 0.0) or 0.0)
        baseline.metric2_std = float(getattr(baseline_metrics, "metric2_std", 0.0) or 0.0)
        # v3 cost: override typical_*_drop with the structural normalizers the
        # 30:30:1 importance weights are designed around. typical_bits = baseline /
        # num_layers ("saving one layer's worth of bits" = bits_norm 1.0);
        # typical_fusion / typical_k = K_LEVELS-derived static maxima.
        baseline.typical_bits_drop = float(
            max(baseline.total_bits_sum / max(int(env.num_layers), 1), 1.0)
        )
        baseline.typical_fusion_count = 12.0
        baseline.typical_k_drop = 5.0

        # baseline 完全 populated 后再校准 reward weights（v3 把 baseline_metric1
        # 和 baseline_metric2 都写进 weights，margin 与阈值都能正确派生）。
        weights = calibrate_weights_from_baseline(baseline)
        env.reward_weights = weights
        status.set_extra("reward_weights", {
            "cost_weight": float(weights.cost_weight),
            "lambda_stab": float(weights.lambda_stab),
            "invalid_penalty": float(weights.invalid_penalty),
            "reward_clip_min": float(weights.reward_clip_min),
            "reward_clip_max": float(weights.reward_clip_max),
            "tier_metric_bonus": float(weights.tier_metric_bonus),
            "tier_stability_bonus": float(weights.tier_stability_bonus),
            "baseline_metric1": float(weights.baseline_metric1),
        })
        log(
            f"  {bullet} 奖励权重（reward weights, v2-style rdv2）："
            f"cost_weight={weights.cost_weight:.4g}, lambda_stab={weights.lambda_stab:.4g}, "
            f"clip=[{weights.reward_clip_min:.1f}, {weights.reward_clip_max:.1f}], "
            f"tier_metric=+{weights.tier_metric_bonus:.1f}, tier_stab=+{weights.tier_stability_bonus:.1f}"
        )

        if not np.isfinite(env.stab_threshold):
            env.stab_threshold = float(baseline.loss_std) * 1.5 + 1e-3
        log(
            f"  {bullet} 基线指标（baseline metrics）：loss_mean={baseline.loss_mean:.4f}, "
            f"loss_std={baseline.loss_std:.4f}, m1={baseline.metric1_mean:.4f}, "
            f"m2={baseline.metric2_mean:.4f}"
        )
        log(
            f"  {bullet} 硬约束阈值: acc_threshold={env.acc_threshold:.4f}, "
            f"stab_threshold={env.stab_threshold:.4f}"
        )
        if not np.isfinite(env.acc_threshold) or env.acc_threshold <= 0.0:
            log(
                f"  {bullet} accuracy threshold will be derived from the all-max BLB "
                "baseline preflight."
            )

        # ---------- 6) Policy + PPO ----------
        torch.manual_seed(int(train_cfg.seed))
        np.random.seed(int(train_cfg.seed) % (2**32))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy = BLBStage2Policy(
            state_dim=int(env.state_dim),
            num_layers=int(env.num_layers),
            per_layer_dims=per_layer_each_dim,
            first_input_levels=5,
        ).to(device)
        baseline_action_vec = np.asarray(_ss_action_vec, dtype=np.int64).reshape(-1)
        if bool(train_cfg.warmstart_baseline_bias):
            try:
                policy.apply_preferred_action_bias(
                    baseline_action_vec,
                    gain=float(train_cfg.warmstart_bias_gain),
                )
                log(
                    f"  {bullet} 策略预热（policy warmstart）：preferred static_skeletons BLB baseline "
                    f"(bias_gain={float(train_cfg.warmstart_bias_gain):.3g})"
                )
            except Exception as exc:
                log(f"  [warmstart][warning] preferred-action bias failed: {exc}")
        optimizer = torch.optim.Adam(policy.parameters(), lr=float(train_cfg.ppo.lr))

        # ---------- 7) Resume（可选） ----------
        start_episode = 0
        update_count = 0
        best_reward = -float("inf")
        best_action_vec: Optional[np.ndarray] = None
        best_breakdown_dict: Optional[Dict[str, Any]] = None
        best_decoded_pickle: Optional[bytes] = None
        best_action_description_paths: Dict[str, str] = {}
        baseline_action_description_paths: Dict[str, str] = {}

        def persist_action_description(label: str, action_vec: np.ndarray) -> Dict[str, str]:
            desc = describe_action_vector(
                action_vec,
                max_sfs=max_sfs,
                num_layers=int(env.num_layers),
                gelu_degree=env.gelu_degree,
                attn_degree=env.attn_degree,
                profile=str(train_cfg.profile),
            )
            paths = write_action_description_files(
                blb_progress_dir,
                desc,
                label=label,
                log_fn=log,
            )
            status.set_extra(f"{label}_action_description", paths)
            return paths

        baseline_action_description_paths = persist_action_description("baseline", baseline_action_vec)
        if baseline_action_description_paths.get("md"):
            log(f"  {bullet} 基线动作可读说明：{baseline_action_description_paths['md']}")

        # Slot-label / kind tables for per-update diagnostics. Built once; the
        # action vector layout is fixed for the entire run.
        baseline_desc = describe_action_vector(
            baseline_action_vec,
            max_sfs=max_sfs,
            num_layers=int(env.num_layers),
            gelu_degree=env.gelu_degree,
            attn_degree=env.attn_degree,
            profile=str(train_cfg.profile),
        )
        baseline_records = list((baseline_desc or {}).get("records") or [])
        slot_label_by_index: List[str] = [
            str(r.get("slot_label", r.get("location", ""))) for r in baseline_records
        ]
        kind_by_index: List[str] = [str(r.get("kind", "")) for r in baseline_records]
        action_dim_by_index = action_dims_for_config(int(env.num_layers))
        action_mask = None
        action_bias = None
        action_mask_meta: Dict[str, Any] = {
            "enabled": False,
            "mode": "none",
            "hash": "",
            "source": "",
            "baseline_logit_bonus": 0.0,
            "allowed_width_min": "",
            "allowed_width_max": "",
        }
        if bool(train_cfg.action_mask_enabled):
            mask_mode = str(train_cfg.action_mask_mode or "none").strip().lower().replace("-", "_")
            if mask_mode in ("", "none", "off", "disabled"):
                action_mask = None
            elif mask_mode == "from_file":
                if not train_cfg.action_mask_file:
                    raise ValueError("action_mask_mode='from_file' requires action_mask_file")
                mask_path = str(train_cfg.action_mask_file)
                if not os.path.isabs(mask_path):
                    mask_path = os.path.join(_resolve_repo_root(), mask_path)
                action_mask, mask_payload = load_action_mask_file(
                    mask_path,
                    expected_width=len(action_dim_by_index),
                    baseline_action=baseline_action_vec.tolist(),
                    action_dims=action_dim_by_index,
                    slot_records=baseline_records,
                )
                action_mask_meta["source"] = str(train_cfg.action_mask_source or mask_path)
                if isinstance(mask_payload, Mapping):
                    action_mask_meta["file_mask_hash"] = str(mask_payload.get("mask_hash", ""))
                    action_mask_meta["file_schema"] = str(mask_payload.get("schema", ""))
            elif mask_mode in ("baseline_only", "near_baseline"):
                action_mask = build_action_mask(
                    num_layers=int(env.num_layers),
                    mode=mask_mode,
                    gelu_degree=env.gelu_degree,
                    attn_degree=env.attn_degree,
                    profile=str(train_cfg.profile),
                    baseline_action=baseline_action_vec.tolist(),
                    max_sfs=max_sfs,
                )
                action_mask_meta["source"] = str(train_cfg.action_mask_source or mask_mode)
            else:
                raise ValueError(f"unknown BLB action mask mode: {train_cfg.action_mask_mode!r}")
            if action_mask is not None:
                ensure_action_allowed(baseline_action_vec.tolist(), action_mask, label="baseline_action")
                action_bias = build_baseline_action_bias(
                    action_dims=action_dim_by_index,
                    baseline_action=baseline_action_vec.tolist(),
                    baseline_logit_bonus=float(train_cfg.action_mask_baseline_logit_bonus),
                )
                allowed_widths = [int(np.asarray(slot, dtype=bool).sum()) for slot in action_mask]
                action_mask_meta.update({
                    "enabled": True,
                    "mode": mask_mode,
                    "hash": action_mask_hash(action_mask),
                    "baseline_logit_bonus": float(train_cfg.action_mask_baseline_logit_bonus),
                    "allowed_width_min": int(min(allowed_widths)) if allowed_widths else 0,
                    "allowed_width_max": int(max(allowed_widths)) if allowed_widths else 0,
                    "allowed_width_mean": float(np.mean(allowed_widths)) if allowed_widths else 0.0,
                    "action_width": int(len(action_dim_by_index)),
                })
                log(
                    f"  {bullet} Action mask 已启用：mode={mask_mode}，"
                    f"hash={action_mask_meta['hash'][:12]}，"
                    f"baseline_logit_bonus={float(train_cfg.action_mask_baseline_logit_bonus):.3g}，"
                    f"allowed_width={action_mask_meta['allowed_width_min']}..{action_mask_meta['allowed_width_max']}"
                )
        status.set_extra("action_mask", action_mask_meta)

        def mutation_diagnostics(action_vec: np.ndarray) -> Dict[str, Any]:
            arr = np.asarray(action_vec, dtype=int).reshape(-1)
            changed = np.flatnonzero(arr != baseline_action_vec)
            by_kind = Counter()
            by_block = Counter()
            effective_count = 0
            ineffective_count = 0
            for idx in changed.tolist():
                if idx < len(baseline_records):
                    by_kind[str(baseline_records[idx].get("kind", ""))] += 1
                    by_block[str(baseline_records[idx].get("block", ""))] += 1
                    if bool(baseline_records[idx].get("effective", True)):
                        effective_count += 1
                    else:
                        ineffective_count += 1
            out = {
                "mutated_slot_count": int(len(changed)),
                "mutated_effective_slot_count": int(effective_count),
                "mutated_ineffective_slot_count": int(ineffective_count),
                "mutated_by_kind": {str(k): int(v) for k, v in sorted(by_kind.items())},
                "mutated_by_block": {str(k): int(v) for k, v in sorted(by_block.items())},
            }
            for kind in ("F", "W", "M", "S", "R", "K"):
                out[f"mutated_{kind}_count"] = int(by_kind.get(kind, 0))
            return out

        mutable_neighbor_indices = [
            idx for idx, record in enumerate(baseline_records)
            if idx < len(action_dim_by_index)
            and bool(record.get("effective", True))
            and str(record.get("block", "")) != "first_input"
            and int(action_dim_by_index[idx]) > 1
        ]
        if action_mask is not None:
            mutable_neighbor_indices = [
                idx for idx in mutable_neighbor_indices
                if any(
                    j != int(baseline_action_vec[idx]) and bool(v)
                    for j, v in enumerate(np.asarray(action_mask[idx], dtype=bool).reshape(-1))
                )
            ]
        cost_probe_specs = [
            ("drop_kind_M", ("M",)),
            ("drop_kind_S", ("S",)),
            ("drop_kind_MS", ("M", "S")),
            ("drop_kind_WMS", ("W", "M", "S")),
        ]
        warmstart_cost_probe_actions: List[Tuple[str, np.ndarray, List[int]]] = []
        for probe_name, probe_kinds in cost_probe_specs:
            probe_action, touched = _build_kind_drop_action(
                baseline_action_vec,
                baseline_records,
                action_dim_by_index,
                kinds=probe_kinds,
                radius=1,
            )
            if touched and not np.array_equal(probe_action, baseline_action_vec):
                if action_mask is None or action_allowed(probe_action.tolist(), action_mask):
                    warmstart_cost_probe_actions.append((probe_name, probe_action, touched))
                else:
                    log(f"  {bullet} 预热成本探针 {probe_name} 不在 action mask 内，已跳过。")
        episode_returns: List[float] = []
        # Initialize curve buffers here (before the resume block) so the
        # resume path can populate them and the later `if not resume` path
        # still sees them as empty lists.
        best_reward_curve: List[float] = []
        ppo_loss_curve: List[float] = []
        if resume_checkpoint_path and os.path.isfile(resume_checkpoint_path):
            try:
                ckpt = self._torch_load_checkpoint(resume_checkpoint_path, map_location=device)
                if isinstance(ckpt, dict):
                    policy.load_state_dict(ckpt["policy"])
                    optimizer.load_state_dict(ckpt["optimizer"])
                    start_episode = int(ckpt.get("completed_episodes", ckpt.get("episode", 0)))
                    update_count = int(ckpt.get("ppo_update_count", 0))
                    episode_returns = [float(x) for x in ckpt.get("episode_returns", [])]
                    # Bug #4 fix: restore curve buffers if present (older
                    # checkpoints may not have them — fall back to empty).
                    saved_best_curve = ckpt.get("best_reward_curve")
                    if saved_best_curve:
                        best_reward_curve = [float(x) for x in saved_best_curve]
                    saved_ppo_curve = ckpt.get("ppo_loss_curve")
                    if saved_ppo_curve:
                        ppo_loss_curve = [float(x) for x in saved_ppo_curve]
                    if "best_reward" in ckpt:
                        best_reward = float(ckpt["best_reward"])
                    if ckpt.get("best_action") is not None:
                        best_action_vec = np.asarray(ckpt["best_action"], dtype=int)
                    best_breakdown_dict = ckpt.get("best_breakdown")
                    best_decoded_pickle = ckpt.get("best_decoded_pickle")
                    rng_state = ckpt.get("rng_state") or {}
                    self._restore_rng_state(rng_state)
                    if best_action_vec is not None and np.isfinite(float(best_reward)):
                        resume_best_episode = ckpt.get("best_episode")
                        if resume_best_episode is None:
                            try:
                                if np.array_equal(best_action_vec, baseline_action_vec):
                                    resume_best_episode = 0
                            except Exception:
                                resume_best_episode = None
                        status.set_best(
                            best_reward=float(best_reward),
                            best_action_vec=best_action_vec,
                            best_breakdown=best_breakdown_dict,
                            best_episode=resume_best_episode,
                        )
                    log(f"  {bullet} 已从 checkpoint 续训（resumed from）：{resume_checkpoint_path}（episode={start_episode}）")
            except Exception as exc:
                log(f"  [resume][警告] 读取 checkpoint 失败：{exc}")

        try:
            env.reset(seed=int(train_cfg.seed))
            _, baseline_reward, _, baseline_info = env.step(baseline_action_vec)
            baseline_breakdown = baseline_info.get("reward_breakdown")
            baseline_breakdown_dict = (
                self._breakdown_to_dict(baseline_breakdown)
                if baseline_breakdown is not None else None
            )
            bm = self._metrics_to_dict(baseline_info.get("metrics"))
            baseline_preflight_metrics = dict(bm)
            if (
                    not bool(baseline_info.get("invalid", False))
                    and not bool(baseline_info.get("apply_failed", False))
                    and not bool(baseline_info.get("eval_failed", False))
            ):
                old_acc_threshold = float(env.acc_threshold)
                acc_threshold_resolution = _baseline_derived_metric_threshold(
                    current_threshold=old_acc_threshold,
                    raw_baseline_metric=baseline.metric1_mean,
                    all_max_blb_metric=bm.get("metric1_mean", 0.0),
                    allowed_drop=getattr(ev, "stage2_limit_tolerance", 0.0),
                )
                env.acc_threshold = float(acc_threshold_resolution.threshold)
                status.set_extra("baseline_accuracy_threshold_calibration", asdict(acc_threshold_resolution))
                if (
                        acc_threshold_resolution.source != "explicit"
                        or abs(float(env.acc_threshold) - old_acc_threshold) > 1e-12
                ):
                    log(
                        f"  {bullet} Baseline accuracy threshold resolved from "
                        f"{old_acc_threshold:.6g} to {float(env.acc_threshold):.6g} "
                        f"using source={acc_threshold_resolution.source}, "
                        f"all_max_blb_metric={acc_threshold_resolution.all_max_blb_metric:.6g}, "
                        f"allowed_drop={acc_threshold_resolution.allowed_drop:.6g}"
                    )
                old_stab_threshold = float(env.stab_threshold)
                new_stab_threshold = _baseline_preflight_stability_threshold(
                    current_threshold=old_stab_threshold,
                    observed_loss_std=bm.get("loss_std", 0.0),
                    tolerance=getattr(ev, "stage2_stability_tolerance", 0.0),
                )
                if (
                        not math.isfinite(old_stab_threshold)
                        or new_stab_threshold > old_stab_threshold + 1e-12
                ):
                    env.stab_threshold = float(new_stab_threshold)
                    baseline_breakdown = compute_reward(
                        baseline_info.get("metrics"),
                        baseline_info.get("opt_signals"),
                        action_avg_k=avg_truncation_k_in_action(
                            baseline_action_vec, env.num_layers,
                        ),
                        baseline=env.baseline,
                        weights=env.reward_weights,
                        acc_threshold=env.acc_threshold,
                        stab_threshold=env.stab_threshold,
                        any_invalid=False,
                    )
                    baseline_reward = float(baseline_breakdown.reward)
                    baseline_info["reward_breakdown"] = baseline_breakdown
                    baseline_breakdown_dict = self._breakdown_to_dict(baseline_breakdown)
                    status.set_extra("baseline_stability_calibration", {
                        "old_stab_threshold": float(old_stab_threshold),
                        "new_stab_threshold": float(env.stab_threshold),
                        "observed_loss_std": float(bm.get("loss_std", 0.0)),
                        "tolerance": float(getattr(ev, "stage2_stability_tolerance", 0.0)),
                        "source": "all_max_blb_preflight",
                    })
                    log(
                        f"  {bullet} Baseline stability threshold calibrated from "
                        f"{old_stab_threshold:.6g} to {float(env.stab_threshold):.6g} "
                        f"using all-max BLB loss_std={float(bm.get('loss_std', 0.0)):.6g}"
                    )
                baseline_breakdown = compute_reward(
                    baseline_info.get("metrics"),
                    baseline_info.get("opt_signals"),
                    action_avg_k=avg_truncation_k_in_action(
                        baseline_action_vec, env.num_layers,
                    ),
                    baseline=env.baseline,
                    weights=env.reward_weights,
                    acc_threshold=env.acc_threshold,
                    stab_threshold=env.stab_threshold,
                    any_invalid=False,
                )
                baseline_reward = float(baseline_breakdown.reward)
                baseline_info["reward_breakdown"] = baseline_breakdown
                baseline_breakdown_dict = self._breakdown_to_dict(baseline_breakdown)
            if is_better_blb_candidate(
                    candidate_reward=float(baseline_reward),
                    candidate_breakdown=baseline_breakdown_dict,
                    best_reward=float(best_reward),
                    best_breakdown=best_breakdown_dict,
            ):
                best_reward = float(baseline_reward)
                best_action_vec = baseline_action_vec.copy()
                best_breakdown_dict = baseline_breakdown_dict
                decoded = baseline_info.get("decoded")
                try:
                    best_decoded_pickle = pickle.dumps(decoded) if decoded is not None else None
                except Exception:
                    best_decoded_pickle = None
                status.set_best(
                    best_reward=best_reward,
                    best_action_vec=best_action_vec,
                    best_breakdown=best_breakdown_dict,
                    best_episode=0,
                )
                best_action_description_paths = persist_action_description("best", best_action_vec)
            status.set_extra("warmstart_baseline_eval", {
                "reward": float(baseline_reward),
                "breakdown": baseline_breakdown_dict,
                "metrics": self._metrics_to_dict(baseline_info.get("metrics")),
                "invalid": bool(baseline_info.get("invalid", False)),
                "apply_failed": bool(baseline_info.get("apply_failed", False)),
                "eval_failed": bool(baseline_info.get("eval_failed", False)),
                "error": str(baseline_info.get("error", "")),
                "optimizer_invalid_summary": str(baseline_info.get("optimizer_invalid_summary", "")),
                "selected_as_incumbent": bool(
                    best_action_vec is not None
                    and np.array_equal(best_action_vec, baseline_action_vec)
                ),
            })
            log(
                f"  {bullet} 基线动作预检（preflight）：reward={float(baseline_reward):+.4f} "
                f"priority={baseline_breakdown_dict.get('priority') if baseline_breakdown_dict else ''} "
                f"invalid={bool(baseline_info.get('invalid', False))} "
                f"loss={float(bm.get('loss_mean', 0.0)):.4f} "
                f"m1={float(bm.get('metric1_mean', 0.0)):.4f} "
                f"m2={float(bm.get('metric2_mean', 0.0)):.4f}"
            )
            if baseline_info.get("error"):
                log(f"  [baseline preflight][error] {baseline_info.get('error')}")
            if baseline_info.get("optimizer_invalid_summary"):
                log(f"  [baseline preflight][optimizer] {baseline_info.get('optimizer_invalid_summary')}")
            if (
                    bool(baseline_info.get("invalid", False))
                    or bool(baseline_info.get("apply_failed", False))
                    or bool(baseline_info.get("eval_failed", False))
                    or bool((baseline_breakdown_dict or {}).get("invalid", False))
                    or int((baseline_breakdown_dict or {}).get("priority", 0)) in (1, 2)
            ):
                raise RuntimeError(
                    "Baseline BLB action preflight failed before RL training; "
                    "the all-max BLB baseline must pass the accuracy and stability thresholds. "
                    f"error={baseline_info.get('error', '')!s}; "
                    f"optimizer_invalid_summary={baseline_info.get('optimizer_invalid_summary', '')!s}; "
                    f"metrics={bm}; "
                    f"breakdown={baseline_breakdown_dict}"
                )
        except Exception as exc:
            log(f"  [warmstart][error] baseline action preflight failed: {exc}")
            raise

        # ---------- 8) 训练循环 ----------
        status.set_phase(f"训练中（PPO 单步 episode，共 {train_cfg.total_episodes} 回合）")
        log("\n训练开始（PPO 单步 episode）...")
        buffer = RolloutBuffer()
        env.reset(seed=int(train_cfg.seed))
        stop_flag_path = None
        graceful_stop_logged = False
        # 训练曲线累积（与 episode_returns 配套；ppo_loss_curve 每次 PPO update 追加）
        # NOTE: these are now initialized earlier (above the resume block) so
        # the resume path can pre-populate them.
        try:
            from noise_rl_module_v2 import (
                NOISE_STAGE_STOP_FLAG_FILENAME,
                consume_stop_flag_file,
                install_graceful_stop_handler,
                is_graceful_stop_requested,
                reset_graceful_stop_state,
                uninstall_graceful_stop_handler,
            )
            stop_flag_path = os.path.join(
                ev.noise_stage_progress_dir, NOISE_STAGE_STOP_FLAG_FILENAME,
            )
            reset_graceful_stop_state()
            consume_stop_flag_file(stop_flag_path)
            install_graceful_stop_handler(log_fn=log)
            log(
                f"  [优雅停止] 训练期间可按 Ctrl+C 或创建 {stop_flag_path} "
                f"触发安全停止（在下一次 PPO rollout 边界保存 checkpoint 后退出）。"
            )
        except Exception as exc:
            uninstall_graceful_stop_handler = None
            is_graceful_stop_requested = None
            consume_stop_flag_file = None
            log(f"  [优雅停止][警告] 无法安装优雅停止处理器，将仅按周期保存 checkpoint：{exc}")

        warmstart_anchor_episodes = _effective_warmstart_anchor_episodes(
            configured_anchor_episodes=train_cfg.warmstart_anchor_episodes,
            rollout_size=int(train_cfg.rollout_size),
            total_episodes=int(train_cfg.total_episodes),
            start_episode=int(start_episode),
            disable_warmstart_on_resume=bool(train_cfg.disable_warmstart_on_resume),
        )
        status.set_extra("warmstart", {
            "baseline_bias": bool(train_cfg.warmstart_baseline_bias),
            "bias_gain": float(train_cfg.warmstart_bias_gain),
            "anchor_episodes": int(warmstart_anchor_episodes),
            "disable_warmstart_on_resume": bool(train_cfg.disable_warmstart_on_resume),
            "neighbor_sampling": bool(train_cfg.warmstart_neighbor_sampling),
            "neighbor_ramp_episodes": int(train_cfg.warmstart_neighbor_ramp_episodes or 0),
            "neighbor_max_mutations": int(train_cfg.warmstart_neighbor_max_mutations),
            "neighbor_max_radius": int(train_cfg.warmstart_neighbor_max_radius),
            "neighbor_mutable_slots": int(len(mutable_neighbor_indices)),
            "guarded_radius2_enabled": bool(train_cfg.guarded_radius2_enabled),
            "guarded_radius2_min_episode": int(train_cfg.guarded_radius2_min_episode),
            "guarded_radius2_stall_window": int(train_cfg.guarded_radius2_stall_window),
            "guarded_radius2_max_mutations": int(train_cfg.guarded_radius2_max_mutations),
            "guarded_radius2_episode_fraction": float(train_cfg.guarded_radius2_episode_fraction),
            "guarded_radius2_cooldown_episodes": int(train_cfg.guarded_radius2_cooldown_episodes),
            "static_invalid_level_mask_enabled": bool(train_cfg.static_invalid_level_mask_enabled),
            "cost_probe_actions": [
                {"name": name, "touched_slots": int(len(touched))}
                for name, _action, touched in warmstart_cost_probe_actions
            ],
        })
        if warmstart_anchor_episodes > 0:
            log(
                f"  {bullet} 预热（warmstart）基线锚点回合数 = {warmstart_anchor_episodes}；"
                "这些早期回合固定使用 all-max BLB 基线动作。"
            )
        if bool(train_cfg.warmstart_neighbor_sampling):
            log(
                f"  {bullet} 预热（warmstart）邻域采样：可变槽位={len(mutable_neighbor_indices)}，"
                f"生效范围=前 {int(train_cfg.warmstart_neighbor_ramp_episodes or 0)} 个绝对回合（absolute episodes），"
                f"单回合最多变更槽位={int(train_cfg.warmstart_neighbor_max_mutations)}，"
                f"最大邻域半径={int(train_cfg.warmstart_neighbor_max_radius)}。"
            )
        if bool(train_cfg.guarded_radius2_enabled):
            log(
                f"  {bullet} 受控 radius2：min_episode={int(train_cfg.guarded_radius2_min_episode)}，"
                f"stall_window={int(train_cfg.guarded_radius2_stall_window)}，"
                f"单回合最多变更槽位={int(train_cfg.guarded_radius2_max_mutations)}，"
                f"采样比例={float(train_cfg.guarded_radius2_episode_fraction):.3g}，"
                f"cooldown={int(train_cfg.guarded_radius2_cooldown_episodes)}。"
            )
        if bool(train_cfg.static_invalid_level_mask_enabled):
            log(
                f"  {bullet} 静态 invalid-level 预裁剪：训练开始前做 baseline-prefix "
                "one-slot Rescale_optimizer 可行性扫描，提前隐藏局部 invalid 的 level。"
            )
        if warmstart_cost_probe_actions:
            log(_format_warmstart_cost_probe_log(warmstart_cost_probe_actions))

        rollout_rewards: List[float] = []
        rollout_metric1: List[float] = []
        rollout_metric2: List[float] = []
        rollout_metric1_min: List[float] = []
        rollout_metric2_min: List[float] = []
        rollout_loss: List[float] = []
        rollout_loss_std: List[float] = []
        rollout_loss_max: List[float] = []
        rollout_priority_counts = {1: 0, 2: 0, 3: 0}
        rollout_invalid_count = 0
        rollout_apply_error_count = 0
        rollout_eval_error_count = 0
        rollout_last_error = ""
        rollout_anchor_count = 0
        rollout_cost_probe_count = 0
        rollout_neighborhood_count = 0
        rollout_policy_count = 0
        rollout_mutated_slot_counts: List[int] = []
        rollout_mutated_effective_slot_counts: List[int] = []
        rollout_mutated_ineffective_slot_counts: List[int] = []
        rollout_mutated_kind_counts = Counter()
        rollout_mutated_block_counts = Counter()

        def mean_or_empty(values: Sequence[float]):
            return float(np.mean(values)) if values else ""

        def min_or_empty(values: Sequence[float]):
            return float(np.min(values)) if values else ""

        def max_or_empty(values: Sequence[float]):
            return float(np.max(values)) if values else ""

        def record_mutation_rollup(mut_diag: Mapping[str, Any]) -> None:
            rollout_mutated_slot_counts.append(int(mut_diag.get("mutated_slot_count", 0)))
            rollout_mutated_effective_slot_counts.append(int(mut_diag.get("mutated_effective_slot_count", 0)))
            rollout_mutated_ineffective_slot_counts.append(int(mut_diag.get("mutated_ineffective_slot_count", 0)))
            for kind, count in dict(mut_diag.get("mutated_by_kind") or {}).items():
                rollout_mutated_kind_counts[str(kind)] += int(count)
            for block, count in dict(mut_diag.get("mutated_by_block") or {}).items():
                rollout_mutated_block_counts[str(block)] += int(count)

        def trace_mask_mutation_fields() -> Dict[str, Any]:
            return {
                "action_source": json.dumps({
                    "anchor": int(rollout_anchor_count),
                    "cost_probe": int(rollout_cost_probe_count),
                    "neighbor": int(rollout_neighborhood_count),
                    "policy": int(rollout_policy_count),
                }, ensure_ascii=False, sort_keys=True),
                "action_mask_mode": str(action_mask_meta.get("mode", "none")),
                "action_mask_hash": str(action_mask_meta.get("hash", "")),
                "action_bias_bonus": float(action_mask_meta.get("baseline_logit_bonus", 0.0) or 0.0),
                "action_source_anchor_count": int(rollout_anchor_count),
                "action_source_cost_probe_count": int(rollout_cost_probe_count),
                "action_source_neighbor_count": int(rollout_neighborhood_count),
                "action_source_policy_count": int(rollout_policy_count),
                "mutated_slot_count": mean_or_empty(rollout_mutated_slot_counts),
                "mutated_effective_slot_count": mean_or_empty(rollout_mutated_effective_slot_counts),
                "mutated_ineffective_slot_count": mean_or_empty(rollout_mutated_ineffective_slot_counts),
                "mutated_slot_count_mean": mean_or_empty(rollout_mutated_slot_counts),
                "mutated_slot_count_max": max_or_empty(rollout_mutated_slot_counts),
                "mutated_effective_slot_count_mean": mean_or_empty(rollout_mutated_effective_slot_counts),
                "mutated_ineffective_slot_count_mean": mean_or_empty(rollout_mutated_ineffective_slot_counts),
                "mutated_F_count": int(rollout_mutated_kind_counts.get("F", 0)),
                "mutated_W_count": int(rollout_mutated_kind_counts.get("W", 0)),
                "mutated_M_count": int(rollout_mutated_kind_counts.get("M", 0)),
                "mutated_S_count": int(rollout_mutated_kind_counts.get("S", 0)),
                "mutated_R_count": int(rollout_mutated_kind_counts.get("R", 0)),
                "mutated_K_count": int(rollout_mutated_kind_counts.get("K", 0)),
                "mutated_by_block": json.dumps(
                    {str(k): int(v) for k, v in sorted(rollout_mutated_block_counts.items())},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }

        def sample_baseline_neighborhood_action(
                obs_t: torch.Tensor,
                episode_offset: int,
                ) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor, int, int]:
            mutation_count, radius = _neighborhood_curriculum(
                episode_offset=int(episode_offset),
                ramp_episodes=int(train_cfg.warmstart_neighbor_ramp_episodes or train_cfg.rollout_size),
                max_mutations=int(train_cfg.warmstart_neighbor_max_mutations),
                max_radius=int(train_cfg.warmstart_neighbor_max_radius),
            )
            candidate = baseline_action_vec.copy()
            if not mutable_neighbor_indices:
                action_for_eval = torch.from_numpy(candidate).long().to(device).unsqueeze(0)
                with torch.no_grad():
                    log_prob_t, _entropy_t, value_t = policy.evaluate_action(
                        obs_t,
                        action_for_eval,
                        action_mask=action_mask,
                        action_bias=action_bias,
                    )
                return candidate, log_prob_t, value_t, 0, radius

            chosen_count = min(int(mutation_count), len(mutable_neighbor_indices))
            chosen = np.random.choice(
                np.asarray(mutable_neighbor_indices, dtype=int),
                size=chosen_count,
                replace=False,
            )
            with torch.no_grad():
                pf = policy.forward(obs_t)
                slot_logits: List[torch.Tensor] = []
                for layer_split in policy._split_layer_logits(pf.layer_logits_flat):
                    slot_logits.extend(layer_split)
                for slot_idx in chosen:
                    dim = int(action_dim_by_index[int(slot_idx)])
                    allowed = _allowed_neighbor_indices(
                        kind=str(kind_by_index[int(slot_idx)]),
                        baseline_idx=int(baseline_action_vec[int(slot_idx)]),
                        dim=dim,
                        radius=int(radius),
                    )
                    allowed = [idx for idx in allowed if 0 <= int(idx) < dim]
                    if action_mask is not None:
                        slot_mask = np.asarray(action_mask[int(slot_idx)], dtype=bool).reshape(-1)
                        allowed = [
                            idx for idx in allowed
                            if int(idx) < int(slot_mask.size) and bool(slot_mask[int(idx)])
                        ]
                    if not allowed:
                        continue
                    non_baseline = [
                        idx for idx in allowed
                        if int(idx) != int(baseline_action_vec[int(slot_idx)])
                    ]
                    # Force most selected slots to actually explore; keeping
                    # the allowed set local is what protects feasibility.
                    pool = non_baseline if non_baseline and np.random.random() < 0.8 else allowed
                    pool_t = torch.tensor(pool, dtype=torch.long, device=device)
                    logits = slot_logits[int(slot_idx)].squeeze(0)[pool_t]
                    local_dist = torch.distributions.Categorical(logits=logits)
                    candidate[int(slot_idx)] = int(pool_t[local_dist.sample()].item())
                action_for_eval = torch.from_numpy(candidate).long().to(device).unsqueeze(0)
                log_prob_t, _entropy_t, value_t = policy.evaluate_action(
                    obs_t,
                    action_for_eval,
                    action_mask=action_mask,
                    action_bias=action_bias,
                )
            return candidate, log_prob_t, value_t, chosen_count, radius

        try:
            for ep in range(start_episode, int(train_cfg.total_episodes)):
                obs = env.reset()

                obs_t = torch.from_numpy(obs).float().to(device).unsqueeze(0)
                warmstart_mode, cost_probe_index = _warmstart_action_mode(
                    episode_index=int(ep),
                    anchor_episodes=int(warmstart_anchor_episodes),
                    cost_probe_count=len(warmstart_cost_probe_actions),
                    neighbor_sampling=bool(train_cfg.warmstart_neighbor_sampling),
                    has_mutable_neighbors=bool(mutable_neighbor_indices),
                    neighbor_ramp_episodes=int(train_cfg.warmstart_neighbor_ramp_episodes or 0),
                )
                episode_offset = int(ep)
                use_anchor_action = warmstart_mode == "anchor"
                use_cost_probe_action = warmstart_mode == "cost_probe"
                use_neighbor_action = warmstart_mode == "neighbor"
                action_source_label = "policy_masked" if action_mask is not None else "policy_unmasked"
                if use_anchor_action:
                    action_source_label = "anchor"
                    action_vec = baseline_action_vec.copy()
                    action_for_eval = torch.from_numpy(action_vec).long().to(device).unsqueeze(0)
                    with torch.no_grad():
                        log_prob_t, _entropy_t, value_t = policy.evaluate_action(
                            obs_t,
                            action_for_eval,
                            action_mask=action_mask,
                            action_bias=action_bias,
                        )
                elif use_cost_probe_action:
                    _probe_name, probe_action_vec, _probe_touched = warmstart_cost_probe_actions[cost_probe_index]
                    action_source_label = f"cost_probe:{_probe_name}"
                    action_vec = probe_action_vec.copy()
                    action_for_eval = torch.from_numpy(action_vec).long().to(device).unsqueeze(0)
                    with torch.no_grad():
                        log_prob_t, _entropy_t, value_t = policy.evaluate_action(
                            obs_t,
                            action_for_eval,
                            action_mask=action_mask,
                            action_bias=action_bias,
                        )
                    rollout_cost_probe_count += 1
                elif use_neighbor_action:
                    action_source_label = "neighbor"
                    action_vec, log_prob_t, value_t, neighbor_mutations, neighbor_radius = (
                        sample_baseline_neighborhood_action(obs_t, episode_offset)
                    )
                    rollout_neighborhood_count += 1
                else:
                    with torch.no_grad():
                        action_t, log_prob_t, value_t = policy.sample_action(
                            obs_t,
                            deterministic=False,
                            action_mask=action_mask,
                            action_bias=action_bias,
                        )
                    action_vec = action_t.squeeze(0).cpu().numpy().astype(np.int64)
                    rollout_policy_count += 1
                log_prob = log_prob_t.detach().reshape(())
                value = value_t.detach().reshape(())
                mut_diag = mutation_diagnostics(action_vec)
                record_mutation_rollup(mut_diag)

                obs_next, reward, done, info = env.step(action_vec)
                breakdown = info.get("reward_breakdown")
                breakdown_dict = self._breakdown_to_dict(breakdown) if breakdown else None

                buffer.add(state=obs, action=action_vec, log_prob=log_prob,
                           reward=float(reward), value=value)
                episode_returns.append(float(reward))
                rollout_rewards.append(float(reward))
                metric_dict = self._metrics_to_dict(info.get("metrics"))
                if metric_dict:
                    if np.isfinite(float(metric_dict.get("metric1_mean", float("nan")))):
                        rollout_metric1.append(float(metric_dict.get("metric1_mean", 0.0)))
                    if np.isfinite(float(metric_dict.get("metric2_mean", float("nan")))):
                        rollout_metric2.append(float(metric_dict.get("metric2_mean", 0.0)))
                    if np.isfinite(float(metric_dict.get("metric1_min", float("nan")))):
                        rollout_metric1_min.append(float(metric_dict.get("metric1_min", 0.0)))
                    if np.isfinite(float(metric_dict.get("metric2_min", float("nan")))):
                        rollout_metric2_min.append(float(metric_dict.get("metric2_min", 0.0)))
                    if np.isfinite(float(metric_dict.get("loss_mean", float("nan")))):
                        rollout_loss.append(float(metric_dict.get("loss_mean", 0.0)))
                    if np.isfinite(float(metric_dict.get("loss_std", float("nan")))):
                        rollout_loss_std.append(float(metric_dict.get("loss_std", 0.0)))
                    if np.isfinite(float(metric_dict.get("loss_max", float("nan")))):
                        rollout_loss_max.append(float(metric_dict.get("loss_max", 0.0)))
                if bool(info.get("apply_failed", False)):
                    rollout_apply_error_count += 1
                if bool(info.get("eval_failed", False)):
                    rollout_eval_error_count += 1
                ep_error_text = ""
                if info.get("error"):
                    ep_error_text = str(info.get("error"))
                    rollout_last_error = ep_error_text
                if breakdown_dict:
                    priority = int(breakdown_dict.get("priority", 0))
                    if priority in rollout_priority_counts:
                        rollout_priority_counts[priority] += 1
                    if bool(breakdown_dict.get("invalid", False)) or bool(info.get("invalid", False)):
                        rollout_invalid_count += 1
                # Bug #7 fix: cap per-rollout main-log error spam at the
                # FIRST 3 errors regardless of kind (invalid_chain + apply +
                # eval). Route every per-episode error/diagnostic to the
                # batched details/ file instead so the main log stays slim.
                main_log_error_count = (
                    int(rollout_invalid_count)
                    + int(rollout_apply_error_count)
                    + int(rollout_eval_error_count)
                )
                if ep_error_text and main_log_error_count <= 3:
                    log(_format_blb_episode_error_log(ep + 1, ep_error_text))
                try:
                    opt_signals_obj = info.get("opt_signals")
                    opt_signals_dict = (
                        {
                            "total_bits_sum": float(getattr(opt_signals_obj, "total_bits_sum", 0)),
                            "total_fusion_count": float(getattr(opt_signals_obj, "total_fusion_count", 0)),
                            "any_invalid": bool(getattr(opt_signals_obj, "any_invalid", False)),
                        }
                        if opt_signals_obj is not None else None
                    )
                    slot_diff_str = ""
                    if best_action_vec is not None:
                        try:
                            slot_diff_str = self._format_action_diff(
                                baseline_action_vec, action_vec, slot_label_by_index, limit=3,
                            )
                        except Exception:
                            slot_diff_str = ""
                    details_writer.append_episode(
                        episode=ep + 1,
                        episode_return=float(reward),
                        priority=int(breakdown_dict.get("priority", 0)) if breakdown_dict else 0,
                        invalid=bool(info.get("invalid", False))
                        or (bool(breakdown_dict.get("invalid", False)) if breakdown_dict else False),
                        error_text=ep_error_text,
                        opt_signals=opt_signals_dict,
                        slot_diff=slot_diff_str,
                        extra_lines=[
                            f"动作来源 action_source={action_source_label}",
                            f"动作掩码 action_mask_mode={action_mask_meta.get('mode', 'none')}",
                            f"动作掩码哈希 action_mask_hash={action_mask_meta.get('hash', '')}",
                            f"baseline logit 加成 action_bias_bonus={action_mask_meta.get('baseline_logit_bonus', 0.0)}",
                            f"变异 slot 数 mutated_slot_count={mut_diag.get('mutated_slot_count', 0)}",
                            f"生效 slot 变异数 mutated_effective_slot_count={mut_diag.get('mutated_effective_slot_count', 0)}",
                            f"无效兼容 slot 变异数 mutated_ineffective_slot_count={mut_diag.get('mutated_ineffective_slot_count', 0)}",
                            "按类型变异 mutated_by_kind="
                            + json.dumps(mut_diag.get("mutated_by_kind", {}), ensure_ascii=False, sort_keys=True),
                            "按 block 变异 mutated_by_block="
                            + json.dumps(mut_diag.get("mutated_by_block", {}), ensure_ascii=False, sort_keys=True),
                        ],
                    )
                except Exception as _details_exc:
                    log(f"  [BLB details][warning] append_episode failed: {_details_exc}")
                if use_anchor_action:
                    rollout_anchor_count += 1

                # 跟踪 best
                if is_better_blb_candidate(
                        candidate_reward=float(reward),
                        candidate_breakdown=breakdown_dict,
                        best_reward=float(best_reward),
                        best_breakdown=best_breakdown_dict,
                ):
                    prev_best_reward = float(best_reward)
                    prev_best_action_vec = best_action_vec.copy() if best_action_vec is not None else None
                    best_reward = float(reward)
                    best_action_vec = action_vec.copy()
                    best_breakdown_dict = breakdown_dict
                    status.set_best(
                        best_reward=best_reward,
                        best_action_vec=best_action_vec,
                        best_breakdown=best_breakdown_dict,
                        best_episode=int(ep + 1),
                    )
                    best_action_description_paths = persist_action_description("best", best_action_vec)
                    # Inline "new best" log with slot-label diff vs previous best
                    # (see CLAUDE.md mental-model #3: never log only the index).
                    try:
                        prev_label = (
                            f"{prev_best_reward:.4f}" if np.isfinite(prev_best_reward) else "-inf"
                        )
                        priority_value = None
                        if isinstance(breakdown_dict, dict):
                            priority_value = int(breakdown_dict.get('priority', 0))
                        if prev_best_action_vec is None:
                            log(
                                _format_blb_best_log(
                                    episode=ep + 1,
                                    best_reward=best_reward,
                                    previous_reward_label=None,
                                    priority=priority_value,
                                )
                            )
                        else:
                            diffs = self._format_action_diff(
                                prev_best_action_vec,
                                best_action_vec,
                                slot_label_by_index,
                                limit=5,
                            )
                            log(
                                _format_blb_best_log(
                                    episode=ep + 1,
                                    best_reward=best_reward,
                                    previous_reward_label=prev_label,
                                    priority=priority_value,
                                    diff_text=diffs,
                                )
                            )
                    except Exception as exc:
                        log(f"  【BLB 新最佳警告】变化位置格式化失败：{exc}")
                    # decoded cfg pickle
                    decoded = info.get("decoded")
                    if decoded is not None:
                        try:
                            best_decoded_pickle = pickle.dumps(decoded)
                        except Exception:
                            best_decoded_pickle = None
                # 累积 best_reward 曲线（每 episode 一个点，等长 episode_returns）
                best_reward_curve.append(float(best_reward) if np.isfinite(best_reward) else 0.0)
                # 状态板：内存更新（不 flush；PPO update 时统一 flush）
                status.update_after_episode(int(ep + 1), float(reward), breakdown=breakdown_dict)

                did_update = False
                # PPO update
                if len(buffer) >= int(train_cfg.rollout_size):
                    metrics = ppo_update(
                        policy,
                        optimizer,
                        buffer,
                        train_cfg.ppo,
                        device,
                        action_mask=action_mask,
                        action_bias=action_bias,
                    )
                    buffer.clear()
                    update_count += 1
                    did_update = True
                    try:
                        ppo_loss_curve.append(float(metrics.get("policy_loss", 0.0)))
                    except Exception:
                        pass
                    status.update_after_ppo_update(int(update_count), metrics)
                    # Inline rollout summary + per-slot-kind entropy. Total
                    # entropy alone hides whether F/W/M/S/R/K slots are
                    # collapsing at uneven rates; surface that here.
                    try:
                        rr = np.asarray(rollout_rewards, dtype=float)
                        raw_ent_by_kind = {}
                        masked_ent_by_kind = {}
                        try:
                            with torch.no_grad():
                                # Use a small slice of the rollout's last
                                # state to keep this cheap (independent of
                                # buffer size — buffer was just cleared).
                                state_t = torch.from_numpy(obs_next).float().to(device).unsqueeze(0)
                                raw_ent_per_dim = policy.per_dim_entropy(state_t).cpu().numpy()
                                masked_ent_per_dim = policy.per_dim_entropy(
                                    state_t,
                                    action_mask=action_mask,
                                    action_bias=action_bias,
                                ).cpu().numpy()
                            raw_ent_by_kind = self._aggregate_entropy_by_kind(raw_ent_per_dim, kind_by_index)
                            masked_ent_by_kind = self._aggregate_entropy_by_kind(masked_ent_per_dim, kind_by_index)
                        except Exception as exc:
                            raw_ent_by_kind = {"计算失败": str(exc)}
                            masked_ent_by_kind = {"计算失败": str(exc)}
                        log(
                            _format_blb_rollout_summary_log(
                                update_count=update_count,
                                episode=ep + 1,
                                total_episodes=train_cfg.total_episodes,
                                reward_mean=float(rr.mean()) if rr.size else 0.0,
                                reward_max=float(rr.max()) if rr.size else 0.0,
                                reward_min=float(rr.min()) if rr.size else 0.0,
                                invalid_count=rollout_invalid_count,
                                priority_counts=rollout_priority_counts,
                                anchor_count=rollout_anchor_count,
                                cost_probe_count=rollout_cost_probe_count,
                                neighborhood_count=rollout_neighborhood_count,
                                policy_loss=float(metrics.get('policy_loss', 0.0)),
                                value_loss=float(metrics.get('value_loss', 0.0)),
                                entropy=float(metrics.get('entropy', 0.0)),
                                clip_fraction=float(metrics.get('clip_fraction', 0.0)),
                                entropy_by_kind=masked_ent_by_kind,
                                raw_entropy_by_kind=raw_ent_by_kind,
                                masked_entropy_by_kind=masked_ent_by_kind,
                            )
                        )
                        # Reward-drop watcher: emit warning.txt entry when
                        # this rollout's mean reward dropped >= threshold
                        # vs. previous rollout. Mirrors legacy v2 warning.txt.
                        try:
                            rollout_start_ep = max(1, int(ep + 1) - len(rollout_rewards) + 1)
                            warn = crash_watcher.observe_rollout(
                                rollout_mean=float(rr.mean()) if rr.size else 0.0,
                                episode_start=int(rollout_start_ep),
                                episode_end=int(ep + 1),
                                details_path=details_writer.current_batch_path,
                            )
                            if warn:
                                log(
                                    f"  【BLB 奖励暴跌】跌幅={float(warn.get('drop', 0.0)):.4f}; "
                                    f"详情见 {crash_watcher.warning_path}"
                                )
                        except Exception as _crash_exc:
                            log(f"  [BLB warning][warning] crash watcher failed: {_crash_exc}")
                    except Exception as exc:
                        log(f"  【BLB Rollout 汇总警告】行内汇总生成失败：{exc}")
                    try:
                        rr = np.asarray(rollout_rewards, dtype=float)
                        trace_path = append_blb_episode_trace_row(
                            blb_progress_dir,
                            {
                                "episode": int(ep + 1),
                                "total_episodes": int(train_cfg.total_episodes),
                                "ppo_update_count": int(update_count),
                                "rollout_reward_mean": float(rr.mean()) if rr.size else 0.0,
                                "rollout_reward_max": float(rr.max()) if rr.size else 0.0,
                                "rollout_reward_min": float(rr.min()) if rr.size else 0.0,
                                "rollout_metric1_mean": mean_or_empty(rollout_metric1),
                                "rollout_metric2_mean": mean_or_empty(rollout_metric2),
                                "rollout_metric1_min": min_or_empty(rollout_metric1_min),
                                "rollout_metric2_min": min_or_empty(rollout_metric2_min),
                                "rollout_loss_mean": mean_or_empty(rollout_loss),
                                "rollout_loss_std_mean": mean_or_empty(rollout_loss_std),
                                "rollout_loss_max": max_or_empty(rollout_loss_max),
                                "best_reward": float(best_reward),
                                "priority1_count": int(rollout_priority_counts.get(1, 0)),
                                "priority2_count": int(rollout_priority_counts.get(2, 0)),
                                "priority3_count": int(rollout_priority_counts.get(3, 0)),
                                "invalid_count": int(rollout_invalid_count),
                                "apply_error_count": int(rollout_apply_error_count),
                                "eval_error_count": int(rollout_eval_error_count),
                                "last_error": rollout_last_error,
                                "anchor_count": int(rollout_anchor_count),
                                "cost_probe_count": int(rollout_cost_probe_count),
                                "policy_loss": float(metrics.get("policy_loss", 0.0)),
                                "value_loss": float(metrics.get("value_loss", 0.0)),
                                "entropy": float(metrics.get("entropy", 0.0)),
                                "raw_entropy_by_kind": json.dumps(
                                    raw_ent_by_kind, ensure_ascii=False, sort_keys=True,
                                ),
                                "masked_entropy_by_kind": json.dumps(
                                    masked_ent_by_kind, ensure_ascii=False, sort_keys=True,
                                ),
                                "clip_fraction": float(metrics.get("clip_fraction", 0.0)),
                                "n_samples": int(metrics.get("n_samples", 0)),
                                **trace_mask_mutation_fields(),
                            },
                            log_fn=log,
                        )
                        if update_count == 1:
                            status.set_extra("episode_trace_csv", trace_path)
                        rollout_rewards = []
                        rollout_metric1 = []
                        rollout_metric2 = []
                        rollout_metric1_min = []
                        rollout_metric2_min = []
                        rollout_loss = []
                        rollout_loss_std = []
                        rollout_loss_max = []
                        rollout_priority_counts = {1: 0, 2: 0, 3: 0}
                        rollout_invalid_count = 0
                        rollout_apply_error_count = 0
                        rollout_eval_error_count = 0
                        rollout_last_error = ""
                        rollout_anchor_count = 0
                        rollout_cost_probe_count = 0
                        rollout_neighborhood_count = 0
                        rollout_policy_count = 0
                        rollout_mutated_slot_counts = []
                        rollout_mutated_effective_slot_counts = []
                        rollout_mutated_ineffective_slot_counts = []
                        rollout_mutated_kind_counts = Counter()
                        rollout_mutated_block_counts = Counter()
                    except Exception as exc:
                        log(f"  [BLB trace][warning] rollout trace update failed: {exc}")
                    if update_count == 1 or update_count % max(1, int(train_cfg.eval_interval // train_cfg.rollout_size)) == 0:
                        self._log_train_iter(
                            log, ep + 1, train_cfg.total_episodes,
                            episode_returns[-train_cfg.rollout_size:], metrics,
                            best_reward,
                        )

                # 周期保存
                if (ep + 1) % max(1, int(train_cfg.save_interval)) == 0:
                    self._save_checkpoint(
                        ev=ev, policy=policy, optimizer=optimizer, episode=ep + 1,
                        best_reward=best_reward, best_action=best_action_vec,
                        best_breakdown=best_breakdown_dict,
                        best_decoded_pickle=best_decoded_pickle,
                        episode_returns=episode_returns,
                        update_count=update_count,
                        fixed_gelu=fixed_gelu,
                        fixed_softmax=fixed_softmax,
                        train_cfg=train_cfg,
                        best_reward_curve=best_reward_curve,
                        ppo_loss_curve=ppo_loss_curve,
                    )

                if (
                        is_graceful_stop_requested is not None
                        and stop_flag_path is not None
                        and is_graceful_stop_requested(stop_flag_path)
                ):
                    if did_update or len(buffer) == 0:
                        live_path = self._save_checkpoint(
                            ev=ev, policy=policy, optimizer=optimizer, episode=ep + 1,
                            best_reward=best_reward, best_action=best_action_vec,
                            best_breakdown=best_breakdown_dict,
                            best_decoded_pickle=best_decoded_pickle,
                            episode_returns=episode_returns,
                            update_count=update_count,
                            fixed_gelu=fixed_gelu,
                            fixed_softmax=fixed_softmax,
                            train_cfg=train_cfg,
                            best_reward_curve=best_reward_curve,
                            ppo_loss_curve=ppo_loss_curve,
                        )
                        if consume_stop_flag_file is not None:
                            consume_stop_flag_file(stop_flag_path)
                        self._mark_stage2_stopped(ev, completed_episodes=ep + 1,
                                                  total_episodes=train_cfg.total_episodes)
                        status.mark_stopped(reason="用户触发优雅停止", completed_episodes=int(ep + 1))
                        try:
                            details_writer.flush()
                        except Exception:
                            pass
                        log(
                            f"  [优雅停止] checkpoint 已写入 → {live_path}\n"
                            f"  下次用相同参数直接运行即可从该 checkpoint 续训练。"
                        )
                        raise SystemExit(0)
                    if not graceful_stop_logged:
                        log(
                            "  [优雅停止] 已收到停止请求；当前 rollout 尚未完成，"
                            "将在下一次 PPO 更新边界保存 checkpoint 后退出。"
                        )
                        graceful_stop_logged = True
        except SystemExit:
            raise  # 优雅停止已经写盘 + 标记，直接出
        except BaseException as exc:
            # 训练崩溃：写崩溃归档（含 traceback + 最后状态），然后再抛
            try:
                snapshot = {
                    "completed_episodes": int(ep + 1) if 'ep' in locals() else 0,
                    "ppo_update_count": int(update_count),
                    "best_reward": float(best_reward) if np.isfinite(best_reward) else None,
                    "len_episode_returns": len(episode_returns),
                    "phase": "训练循环崩溃",
                }
                err_path = dump_crash_report(blb_progress_dir, exc=exc, last_state=snapshot, log_fn=log)
                if err_path:
                    log(f"  [BLB崩溃归档] traceback 已写入 → {err_path}")
                status.set_phase("崩溃")
                status.set_extra("crash_summary", {"type": type(exc).__name__, "msg": str(exc)})
            except Exception:
                pass
            raise
        finally:
            if uninstall_graceful_stop_handler is not None:
                uninstall_graceful_stop_handler()

        # 残留 buffer flush
        if len(buffer) > 0:
            metrics = ppo_update(
                policy,
                optimizer,
                buffer,
                train_cfg.ppo,
                device,
                action_mask=action_mask,
                action_bias=action_bias,
            )
            buffer.clear()
            update_count += 1
            try:
                ppo_loss_curve.append(float(metrics.get("policy_loss", 0.0)))
            except Exception:
                pass
            status.update_after_ppo_update(int(update_count), metrics)
            try:
                rr = np.asarray(rollout_rewards, dtype=float)
                trace_path = append_blb_episode_trace_row(
                    blb_progress_dir,
                    {
                        "episode": int(train_cfg.total_episodes),
                        "total_episodes": int(train_cfg.total_episodes),
                        "ppo_update_count": int(update_count),
                        "rollout_reward_mean": float(rr.mean()) if rr.size else 0.0,
                        "rollout_reward_max": float(rr.max()) if rr.size else 0.0,
                        "rollout_reward_min": float(rr.min()) if rr.size else 0.0,
                        "rollout_metric1_mean": mean_or_empty(rollout_metric1),
                        "rollout_metric2_mean": mean_or_empty(rollout_metric2),
                        "rollout_metric1_min": min_or_empty(rollout_metric1_min),
                        "rollout_metric2_min": min_or_empty(rollout_metric2_min),
                        "rollout_loss_mean": mean_or_empty(rollout_loss),
                        "rollout_loss_std_mean": mean_or_empty(rollout_loss_std),
                        "rollout_loss_max": max_or_empty(rollout_loss_max),
                        "best_reward": float(best_reward),
                        "priority1_count": int(rollout_priority_counts.get(1, 0)),
                        "priority2_count": int(rollout_priority_counts.get(2, 0)),
                        "priority3_count": int(rollout_priority_counts.get(3, 0)),
                        "invalid_count": int(rollout_invalid_count),
                        "apply_error_count": int(rollout_apply_error_count),
                        "eval_error_count": int(rollout_eval_error_count),
                        "last_error": rollout_last_error,
                        "anchor_count": int(rollout_anchor_count),
                        "cost_probe_count": int(rollout_cost_probe_count),
                        "policy_loss": float(metrics.get("policy_loss", 0.0)),
                        "value_loss": float(metrics.get("value_loss", 0.0)),
                        "entropy": float(metrics.get("entropy", 0.0)),
                        "clip_fraction": float(metrics.get("clip_fraction", 0.0)),
                        "n_samples": int(metrics.get("n_samples", 0)),
                        **trace_mask_mutation_fields(),
                    },
                    log_fn=log,
                )
                status.set_extra("episode_trace_csv", trace_path)
            except Exception as exc:
                log(f"  [BLB trace][warning] final rollout trace update failed: {exc}")

        # ---------- 9) Final 落盘 ----------
        final_save_path = self._save_checkpoint(
            ev=ev, policy=policy, optimizer=optimizer,
            episode=int(train_cfg.total_episodes),
            best_reward=best_reward, best_action=best_action_vec, label="final",
            best_breakdown=best_breakdown_dict,
            best_decoded_pickle=best_decoded_pickle,
            episode_returns=episode_returns,
            update_count=update_count,
            fixed_gelu=fixed_gelu,
            fixed_softmax=fixed_softmax,
            train_cfg=train_cfg,
            best_reward_curve=best_reward_curve,
            ppo_loss_curve=ppo_loss_curve,
        )
        # Flush any pending step-detail buffer so the last (possibly partial)
        # batch lands on disk before final report is written.
        try:
            flushed_path = details_writer.flush()
            if flushed_path:
                log(f"  {bullet} 训练末批 details 已刷盘 → {flushed_path}")
        except Exception as _flush_exc:
            log(f"  [BLB details][warning] final flush failed: {_flush_exc}")
        log(f"\n训练完成：best_reward={best_reward:.4f}")
        log(f"  {bullet} Final policy 已保存到：{final_save_path}")
        if int(crash_watcher.total_count) > 0:
            log(
                f"  {bullet} 共记录 {int(crash_watcher.total_count)} 条奖励暴跌警告 → "
                f"{crash_watcher.warning_path}"
            )

        if best_action_vec is not None:
            if not best_action_description_paths:
                best_action_description_paths = persist_action_description("best", best_action_vec)
            blb_cfg_dump_path = os.path.join(
                ev.noise_stage_progress_dir, BLB_STAGE2_BEST_CFG_FILENAME,
            )
            try:
                os.makedirs(os.path.dirname(blb_cfg_dump_path), exist_ok=True)
                with open(blb_cfg_dump_path, "wb") as f:
                    pickle.dump({
                        "best_action_vec": best_action_vec,
                        "best_decoded_pickle": best_decoded_pickle,
                        "best_reward": float(best_reward),
                        "best_breakdown": best_breakdown_dict,
                        "profile": train_cfg.profile,
                        "num_layers": int(ev.total_layers),
                        "best_action_description_paths": dict(best_action_description_paths or {}),
                    }, f)
                log(f"  {bullet} Best BLB cfg 已保存到：{blb_cfg_dump_path}")
            except Exception as exc:
                log(f"  [警告] 无法保存 best_blb_cfg：{exc}")

        # ---------- 9.1) 训练曲线 + 最终报告 ----------
        status.set_phase("写训练曲线 / 最终报告")
        try:
            curve_paths = write_training_curves(
                blb_progress_dir,
                episode_returns=episode_returns,
                best_reward_curve=best_reward_curve,
                ppo_loss_curve=ppo_loss_curve,
                log_fn=log,
            )
            if curve_paths.get("png"):
                log(f"  {bullet} 训练曲线 PNG → {curve_paths['png']}")
            if curve_paths.get("npz"):
                log(f"  {bullet} 训练曲线 NPZ → {curve_paths['npz']}")
        except Exception as exc:
            log(f"  [警告] 写训练曲线失败：{exc}")

        try:
            report_path = write_blb_final_report(
                blb_progress_dir,
                run_basename=run_basename,
                profile=str(train_cfg.profile),
                total_episodes=int(train_cfg.total_episodes),
                completed_episodes=int(train_cfg.total_episodes),
                elapsed_sec=float(time.time() - status._t0),
                best_reward=float(best_reward) if np.isfinite(best_reward) else 0.0,
                best_breakdown=best_breakdown_dict,
                best_action_vec=(best_action_vec.tolist() if best_action_vec is not None else None),
                baseline={
                    "total_bits_sum": int(baseline.total_bits_sum),
                    "total_fusion_count": int(baseline.total_fusion_count),
                    "avg_k": float(baseline.avg_k),
                    "loss_mean": float(getattr(baseline, "loss_mean", 0.0)),
                    "metric1_mean": float(getattr(baseline, "metric1_mean", 0.0)),
                },
                reward_weights={
                    "design": "v2-style rdv2",
                    "cost_weight": float(weights.cost_weight),
                    "lambda_stab": float(weights.lambda_stab),
                    "invalid_penalty": float(weights.invalid_penalty),
                    "reward_clip_min": float(weights.reward_clip_min),
                    "reward_clip_max": float(weights.reward_clip_max),
                    "tier_metric_bonus": float(weights.tier_metric_bonus),
                    "tier_stability_bonus": float(weights.tier_stability_bonus),
                    "baseline_metric1": float(weights.baseline_metric1),
                    "acc_threshold": float(env.acc_threshold),
                    "stab_threshold": float(env.stab_threshold),
                },
                episode_returns=episode_returns,
                rescale_invoker_kind="in_process_real",
                extra_lines=[
                    f"Warmstart baseline bias: {bool(train_cfg.warmstart_baseline_bias)}",
                    f"Warmstart anchor episodes: {int(train_cfg.warmstart_anchor_episodes or 0)}",
                    f"Action mask enabled: {bool(action_mask_meta.get('enabled', False))}",
                    f"Action mask mode: {action_mask_meta.get('mode', 'none')}",
                    f"Action mask hash: {action_mask_meta.get('hash', '')}",
                    f"Action mask source: {action_mask_meta.get('source', '')}",
                    f"Baseline action logit bonus: {action_mask_meta.get('baseline_logit_bonus', 0.0)}",
                    f"Rollout trace CSV: {os.path.join(blb_progress_dir, 'blb_stage2_episode_trace.csv')}",
                    f"Baseline action readable JSON: {baseline_action_description_paths.get('json', '')}",
                    f"Baseline action readable Markdown: {baseline_action_description_paths.get('md', '')}",
                    f"Best action readable JSON: {best_action_description_paths.get('json', '')}",
                    f"Best action readable Markdown: {best_action_description_paths.get('md', '')}",
                ],
                log_fn=log,
            )
            log(f"  {bullet} 最终训练报告 → {report_path}")
            status.set_extra("final_report_path", report_path)
        except Exception as exc:
            log(f"  [警告] 写最终报告失败：{exc}")
        status.set_phase("已完成")

        # 2026-06-01 解耦：stage2-only 完成 -> 归档进 stage2/record/ 并打 COMPLETED（best-effort，
        # 任何异常只记日志，绝不让训练在收尾处崩溃）。一个 stage2 record 严格绑定其前置 stage1。
        if getattr(ev, "decoupled_layout", False) and best_action_vec is not None:
            try:
                import datetime as _dt
                from config import run_layout as _rl
                _wd = os.path.normpath(str(getattr(ev, "run_output_dir", "") or ""))   # <root>/stage2/<combo>
                if _wd and _wd != ".":
                    _combo = os.path.basename(_wd)
                    _root = os.path.dirname(os.path.dirname(_wd))
                    _bd = best_breakdown_dict if isinstance(best_breakdown_dict, dict) else {}

                    def _g(*keys):
                        for _k in keys:
                            if _k in _bd and _bd[_k] is not None:
                                try:
                                    return float(_bd[_k])
                                except Exception:
                                    return _bd[_k]
                        return None

                    _paths = best_action_description_paths or {}
                    _final_config = {
                        "stage": 2,
                        "combo": _combo,
                        "profile": str(train_cfg.profile),
                        "num_layers": int(ev.total_layers),
                        "blb_v3_best_action_vec": best_action_vec.tolist(),
                        # 前置 Stage-1（一个 stage2 严格绑定一个 stage1）。
                        "gelu_degree_per_layer": np.asarray(fixed_gelu, dtype=int).tolist(),
                        "softmax_degree_per_layer": np.asarray(fixed_softmax, dtype=int).tolist(),
                        "best_action_readable_json": _paths.get("json", ""),
                        "best_action_readable_md": _paths.get("md", ""),
                    }
                    _final_eval = {
                        "source": "training_best_mean_of_K_trials",
                        "note": "basic snapshot (训练记录的 K 次 MC 噪声 trial 最优档); "
                                "重型同-cost 组对比见独立 final-eval 工具。",
                        "best_reward": float(best_reward) if np.isfinite(best_reward) else None,
                        "loss": _g("loss_mean", "loss"),
                        "metric1": _g("metric1_mean", "metric1"),
                        "metric2": _g("metric2_mean", "metric2"),
                        "cost": {
                            "total_bits_sum": _g("total_bits_sum", "total_bits"),
                            "total_fusion_count": _g("total_fusion_count", "fusion_count"),
                            "sum_truncation_k": _g("sum_truncation_k", "sum_k"),
                            "avg_k": _g("avg_k"),
                        },
                        "baseline_cost": {
                            "total_bits_sum": int(baseline.total_bits_sum),
                            "total_fusion_count": int(baseline.total_fusion_count),
                            "avg_k": float(baseline.avg_k),
                            "loss_mean": float(getattr(baseline, "loss_mean", 0.0)),
                            "metric1_mean": float(getattr(baseline, "metric1_mean", 0.0)),
                        },
                        "breakdown": _bd,
                    }
                    _metadata = {
                        "stage": 2,
                        "combo": _combo,
                        "profile": str(train_cfg.profile),
                        "data_path": getattr(ev, "data_path", ""),
                        "completed_at": _dt.datetime.now().isoformat(),
                        "episodes": int(train_cfg.total_episodes),
                        "stage1_run_id": getattr(ev, "stage1_run_id", ""),
                        "stage2_limit_tolerance": getattr(ev, "stage2_limit_tolerance", None),
                        "stage2_stability_tolerance": getattr(ev, "stage2_stability_tolerance", None),
                    }
                    _report_md = (
                        f"# Stage-2 record: {_combo}\n\n"
                        f"- profile: {train_cfg.profile}, num_layers: {ev.total_layers}\n"
                        f"- best_reward: {best_reward}\n"
                        f"- prerequisite Stage-1 gelu: {np.asarray(fixed_gelu, dtype=int).tolist()}\n"
                        f"- prerequisite Stage-1 softmax: {np.asarray(fixed_softmax, dtype=int).tolist()}\n"
                        f"- best action readable: {_paths.get('md', '')}\n"
                    )
                    _curves = [
                        os.path.join(blb_progress_dir, "blb_stage2_training_curve.png"),
                        _paths.get("json", ""),
                        _paths.get("md", ""),
                    ]
                    _rdir, _rid, _n = _rl.snapshot_decoupled_record(
                        2, _combo, _wd,
                        final_config=_final_config,
                        final_eval=_final_eval,
                        metadata=_metadata,
                        curve_paths=[p for p in _curves if p],
                        report_md=_report_md,
                        root=_root,
                    )
                    log(f"  {bullet} [解耦] Stage-2 已归档进 record → {_rdir}（COMPLETED 已标记）")
            except Exception as _snap_exc:
                log(f"  [解耦][警告] Stage-2 record 归档失败（不影响训练结果）：{_snap_exc}")

        # ---------- 10) 还原模型到干净的多项式近似态（不带 BLB 噪声） ----------
        try:
            for restore_name in (
                    "restore_layer_block5_noise", "restore_layer_block4_noise",
                    "restore_layer_block3_noise", "restore_layer_block2_noise",
                    "restore_layer_block1_noise", "restore_blb_first_input_noise",
            ):
                method = getattr(ev.reversible_handler, restore_name, None)
                if method is None:
                    continue
                try:
                    method(layer_indices=list(range(ev.total_layers)))
                except Exception:
                    pass
        finally:
            ev.apply_configuration(fixed_gelu, fixed_softmax)

        # ---------- 11) 构造与旧版兼容的返回 dict ----------
        cost_reference_noise_config = ev._get_max_noise_configuration()
        cost_reference_tot_c, _ = ev.get_noise_simulated_cost(**cost_reference_noise_config)
        legacy_best = _build_legacy_compatible_best_noise_config(ev)

        # NOTE: ``best_noise_config`` below is a legacy-shape all-max baseline
        # for backwards compatibility with ``UnifiedFinalEvaluationModule`` only.
        # The BLB-aware final-eval path (``Paean/blb_action_eval.py`` ->
        # ``BLBActionFinalEvaluationModule``) reads ``blb_v3_best_action_vec``
        # from the result dict and installs the real BLB best via
        # ``BLBNoiseRLBridge.apply``. ``layer_importance_evaluator
        # ._should_run_blb_action_final_eval`` selects that path when the BLB
        # best action is present, so legacy ``best_noise_config`` is never the
        # actual final-eval target — it is purely a compat-shape placeholder.
        log(
            "  [BLB final_eval] result['blb_v3_best_action_vec'] -> "
            "BLBActionFinalEvaluationModule will install via bridge.apply. "
            "result['best_noise_config'] = all-max compat baseline (unused for BLB best)."
        )

        # 用 baseline 来推算 limit_loss / limit_p / limit_s（与旧版一致）
        base_loss, base_p, base_s, _ = ev.evaluate_model(
            fixed_gelu, fixed_softmax, use_train=False,
            split=ev.get_reward_reference_split_name(),
        )
        limit_dict = ev.build_constraint_limits_from_metrics(base_loss, base_p, base_s)
        limit_loss = float(limit_dict["loss"])
        limit_p = float(limit_dict["metric1"])
        limit_s = float(limit_dict["metric2"])
        stage2_metric_allowed_drop = max(
            0.0, _selection_float(getattr(ev, "stage2_limit_tolerance", 0.0), 0.0),
        )
        if baseline_preflight_metrics:
            preflight_loss = _selection_float(baseline_preflight_metrics.get("loss_mean"), base_loss)
            preflight_metric1 = _selection_float(baseline_preflight_metrics.get("metric1_mean"), base_p)
            preflight_metric2 = _selection_float(baseline_preflight_metrics.get("metric2_mean"), base_s)
            limit_loss = float(preflight_loss * (1.0 + stage2_metric_allowed_drop))
            if acc_threshold_resolution is not None:
                limit_p = float(acc_threshold_resolution.threshold)
            else:
                limit_p = max(
                    0.0,
                    float(preflight_metric1) * (1.0 - stage2_metric_allowed_drop),
                )
            limit_s = max(
                0.0,
                float(preflight_metric2) * (1.0 - stage2_metric_allowed_drop),
            )
            limit_dict = {
                "loss": float(limit_loss),
                "metric1": float(limit_p),
                "metric2": float(limit_s),
            }

        result: Dict[str, Any] = {
            "fixed_gelu": fixed_gelu.copy(),
            "fixed_softmax": fixed_softmax.copy(),
            "baseline_noise_config": {k: v.copy() for k, v in cost_reference_noise_config.items()},
            "baseline_tot_c": float(cost_reference_tot_c),
            "cost_reference_noise_config": {k: v.copy() for k, v in cost_reference_noise_config.items()},
            "cost_reference_source": "max_noise_configuration",
            "performance_baseline_gelu": fixed_gelu.copy(),
            "performance_baseline_softmax": fixed_softmax.copy(),
            "performance_baseline_source": "stage1_fixed_low_risk_noise",
            "k_trials": int(train_cfg.num_trials_per_step),
            "probe_size": int(getattr(ev, "stage2_probe_size", 256)),
            "limit_computation_method": (
                "all_max_blb_baseline_relative_tolerance"
                if baseline_preflight_metrics else "raw_baseline_tolerance_percentage"
            ),
            "limit_tolerance": float(getattr(ev, "stage2_limit_tolerance", 0.05)),
            "stability_tolerance": float(getattr(ev, "stage2_stability_tolerance", 0.05)),
            "threshold_source": (
                acc_threshold_resolution.source
                if acc_threshold_resolution is not None else "unknown"
            ),
            "threshold_baseline_source": (
                "all_max_blb_preflight" if baseline_preflight_metrics else "raw_model_fallback"
            ),
            "threshold_allowed_drop": float(stage2_metric_allowed_drop),
            "threshold_tolerance": float(stage2_metric_allowed_drop),
            "raw_model_baseline_metrics": {
                "loss": float(base_loss),
                "metric1": float(base_p),
                "metric2": float(base_s),
            },
            "all_max_blb_baseline_metrics": dict(baseline_preflight_metrics),
            "search_limits": {"loss": float(limit_loss),
                              "metric1": float(limit_p),
                              "metric2": float(limit_s)},
            "status": "completed",
            # legacy 兼容字段：让 final-eval 能像 baseline 一样跑（BLB 真正的 best 配置在
            # blb_* 字段里）。
            "best_noise_config": {k: v.copy() for k, v in legacy_best.items()},
            "stable_search_best_noise_config": {k: v.copy() for k, v in legacy_best.items()},
            "stable_joint_best_noise_config": {k: v.copy() for k, v in legacy_best.items()},
            "selection_diagnostics": {
                "selection_mode": "blb_v3_runtime_best",
                "best_reward": float(best_reward),
                "best_breakdown": best_breakdown_dict,
                "k_trials": int(train_cfg.num_trials_per_step),
                "probe_size": int(getattr(ev, "stage2_probe_size", 256)),
                "baseline_action_description_paths": dict(baseline_action_description_paths or {}),
                "best_action_description_paths": dict(best_action_description_paths or {}),
                "action_mask": dict(action_mask_meta),
            },
            "shortlist_diagnostics": {},
            "limit_loss": float(limit_loss),
            "limit_p": float(limit_p),
            "limit_s": float(limit_s),
            "proxy_limit_loss": float(limit_loss),
            "proxy_limit_p": float(limit_p),
            "proxy_limit_s": float(limit_s),
            "proxy_base_loss": float(base_loss),
            "proxy_base_p": float(base_p),
            "proxy_base_s": float(base_s),
            "training_eval_split": str(ev.get_reward_reference_split_name()),
            "training_hparams": {
                "blb_v3_total_episodes": int(train_cfg.total_episodes),
                "blb_v3_rollout_size": int(train_cfg.rollout_size),
                "blb_v3_ppo_lr": float(train_cfg.ppo.lr),
                "blb_v3_clip_range": float(train_cfg.ppo.clip_range),
                "blb_v3_n_epochs": int(train_cfg.ppo.n_epochs),
                "blb_v3_minibatch_size": int(train_cfg.ppo.minibatch_size),
                "blb_v3_ent_coef": float(train_cfg.ppo.ent_coef),
                "blb_v3_value_coef": float(train_cfg.ppo.value_coef),
                "blb_v3_max_grad_norm": float(train_cfg.ppo.max_grad_norm),
                "blb_v3_acc_threshold": float(env.acc_threshold),
                "blb_v3_stab_threshold": float(env.stab_threshold),
                "blb_v3_acc_threshold_source": (
                    acc_threshold_resolution.source
                    if acc_threshold_resolution is not None else "unknown"
                ),
                "blb_v3_threshold_baseline_source": (
                    "all_max_blb_preflight" if baseline_preflight_metrics else "raw_model_fallback"
                ),
                "blb_v3_threshold_allowed_drop": float(stage2_metric_allowed_drop),
                "blb_v3_threshold_tolerance": float(stage2_metric_allowed_drop),
                "blb_v3_reward_design": str(getattr(weights, "reward_design", "stage1_aligned")),
                "blb_v3_cost_weight": float(weights.cost_weight),
                "blb_v3_lambda_stab": float(weights.lambda_stab),
                "blb_v3_invalid_penalty": float(weights.invalid_penalty),
                "blb_v3_tier_metric_bonus": float(weights.tier_metric_bonus),
                "blb_v3_tier_stability_bonus": float(weights.tier_stability_bonus),
                "blb_v3_warmstart_baseline_bias": bool(train_cfg.warmstart_baseline_bias),
                "blb_v3_warmstart_bias_gain": float(train_cfg.warmstart_bias_gain),
                "blb_v3_warmstart_anchor_episodes": int(train_cfg.warmstart_anchor_episodes or 0),
                "blb_v3_disable_warmstart_on_resume": bool(train_cfg.disable_warmstart_on_resume),
                "blb_v3_warmstart_neighbor_sampling": bool(train_cfg.warmstart_neighbor_sampling),
                "blb_v3_warmstart_neighbor_ramp_episodes": int(train_cfg.warmstart_neighbor_ramp_episodes or 0),
                "blb_v3_warmstart_neighbor_max_mutations": int(train_cfg.warmstart_neighbor_max_mutations),
                "blb_v3_warmstart_neighbor_max_radius": int(train_cfg.warmstart_neighbor_max_radius),
                "blb_v3_guarded_radius2_enabled": bool(train_cfg.guarded_radius2_enabled),
                "blb_v3_guarded_radius2_min_episode": int(train_cfg.guarded_radius2_min_episode),
                "blb_v3_guarded_radius2_stall_window": int(train_cfg.guarded_radius2_stall_window),
                "blb_v3_guarded_radius2_max_mutations": int(train_cfg.guarded_radius2_max_mutations),
                "blb_v3_guarded_radius2_episode_fraction": float(train_cfg.guarded_radius2_episode_fraction),
                "blb_v3_guarded_radius2_cooldown_episodes": int(train_cfg.guarded_radius2_cooldown_episodes),
                "blb_v3_static_invalid_level_mask_enabled": bool(train_cfg.static_invalid_level_mask_enabled),
                "blb_v3_action_mask_enabled": bool(action_mask_meta.get("enabled", False)),
                "blb_v3_action_mask_mode": str(action_mask_meta.get("mode", "none")),
                "blb_v3_action_mask_hash": str(action_mask_meta.get("hash", "")),
                "blb_v3_action_mask_source": str(action_mask_meta.get("source", "")),
                "blb_v3_action_mask_baseline_logit_bonus": float(
                    action_mask_meta.get("baseline_logit_bonus", 0.0) or 0.0
                ),
                "k_trials": int(train_cfg.num_trials_per_step),
                "probe_size": int(getattr(ev, "stage2_probe_size", 256)),
            },
            "reward_diagnostics": {
                "terminal_reward_mode": "blb_v3_three_priority",
                "k_trials": int(train_cfg.num_trials_per_step),
                "probe_size": int(getattr(ev, "stage2_probe_size", 256)),
                "episode_return_mean": float(np.mean(episode_returns)) if episode_returns else None,
                "episode_return_max": float(np.max(episode_returns)) if episode_returns else None,
                "best_reward": float(best_reward),
            },
            # ---------- 新版独有字段 ----------
            "blb_v3_best_action_vec": (
                best_action_vec.tolist() if best_action_vec is not None else None
            ),
            "blb_v3_best_reward": float(best_reward),
            "blb_v3_best_action_description_paths": dict(best_action_description_paths or {}),
            "blb_v3_baseline_action_description_paths": dict(baseline_action_description_paths or {}),
            "blb_v3_profile": str(train_cfg.profile),
            "blb_v3_total_episodes": int(train_cfg.total_episodes),
            "blb_v3_action_mask": dict(action_mask_meta),
            "rl_variant": "blb_v3",
        }
        return result

    # ------------------------------------------------------------------
    # 配置解析
    # ------------------------------------------------------------------
    def _build_train_config_from_evaluator(self, ev) -> BLBStage2TrainConfig:
        cfg = BLBStage2TrainConfig()
        # 1) total_episodes：复用 evaluator.stage2_rl_episodes
        try:
            cfg.total_episodes = int(getattr(ev, "stage2_rl_episodes", cfg.total_episodes))
        except Exception:
            pass
        # 2) PPO LR：复用 evaluator.stage2_ppo_lr_initial
        try:
            cfg.ppo.lr = float(getattr(ev, "stage2_ppo_lr_initial", cfg.ppo.lr))
        except Exception:
            pass
        # 3) profile：从 dataset_key（数据集名）推断
        try:
            cfg.profile = str(ev.dataset_key)
        except Exception:
            pass
        # 4) num_trials_per_step / probe_batch_count
        try:
            cfg.num_trials_per_step = int(getattr(ev, "stage2_k_trials", cfg.num_trials_per_step))
        except Exception:
            pass
        # 4b) reward_devices: --blb-v3-reward-devices "0,1" arrives as a string
        # on the evaluator. Parse here so train_cfg holds a List[int]; empty /
        # 1-device → ProbeRunner stays disabled.
        from .probe_runner import parse_device_ids
        spec = getattr(ev, "blb_v3_reward_devices", None)
        parsed = parse_device_ids(spec)
        if parsed:
            cfg.reward_devices = parsed
        # 4c) stage2_rl_devices: --stage2-rl-devices "0,1,2,3,4" → episode-
        # parallel rollout (fusion mode). Same Fire string/tuple parsing.
        spec2 = getattr(ev, "stage2_rl_devices", None)
        parsed2 = parse_device_ids(spec2)
        if parsed2:
            cfg.stage2_rl_devices = parsed2
        v = getattr(ev, "stage2_workers_per_device", None)
        if v not in (None, ""):
            try:
                cfg.stage2_workers_per_device = max(1, int(v))
            except Exception:
                pass
        # 5) BLB v3 always uses the real in-process Rescale_optimizer.  Legacy
        # invoker selection attributes are deliberately ignored.
        root = getattr(ev, "blb_v3_inproc_rescale_optimizer_root", None)
        if root not in (None, ""):
            cfg.inproc_rescale_optimizer_root = str(root)
        # 6) rollout_size / save_interval / eval_interval：直接从 evaluator 取（如果有挂）
        for cfg_field, attr_name in (
                ("rollout_size", "blb_v3_rollout_size"),
                ("save_interval", "blb_v3_save_interval"),
                ("eval_interval", "blb_v3_eval_interval"),
                ("calibrate_baseline_samples", "blb_v3_calibrate_baseline_samples"),
                ("warmstart_anchor_episodes", "blb_v3_warmstart_anchor_episodes"),
                ("warmstart_neighbor_ramp_episodes", "blb_v3_warmstart_neighbor_ramp_episodes"),
                ("warmstart_neighbor_max_mutations", "blb_v3_warmstart_neighbor_max_mutations"),
                ("warmstart_neighbor_max_radius", "blb_v3_warmstart_neighbor_max_radius"),
                ("guarded_radius2_min_episode", "blb_v3_guarded_radius2_min_episode"),
                ("guarded_radius2_stall_window", "blb_v3_guarded_radius2_stall_window"),
                ("guarded_radius2_max_mutations", "blb_v3_guarded_radius2_max_mutations"),
                ("guarded_radius2_cooldown_episodes", "blb_v3_guarded_radius2_cooldown_episodes"),
                ("ent_coef_ramp_episodes", "blb_v3_ent_coef_ramp_episodes"),
                ("online_num_trials_per_step", "blb_v3_online_k_trials"),
                ("terminal_eval_batch_size", "blb_v3_terminal_eval_batch_size"),
                ("promotion_validation_trials", "blb_v3_promotion_validation_trials"),
                ("final_selection_top_n", "blb_v3_final_selection_top_n"),
                ("final_selection_validation_trials", "blb_v3_final_selection_validation_trials"),
                ("baseline_groups", "blb_v3_baseline_groups"),
                ("baseline_trials_per_group", "blb_v3_baseline_trials_per_group"),
                ("constraint_bootstrap_samples", "blb_v3_constraint_bootstrap_samples"),
                ("seed", "final_eval_random_seed"),
        ):
            v = getattr(ev, attr_name, None)
            if v is None:
                continue
            try:
                setattr(cfg, cfg_field, int(v))
            except Exception:
                pass
        # PPO-only guard. Keep this explicit so evaluator/preset values cannot
        # re-enable the removed GRPO path.
        _rl_algo = getattr(ev, "rl_algo", None)
        if _rl_algo not in (None, ""):
            cfg.rl_algo = _normalize_supported_rl_algo(
                _rl_algo, context="evaluator.rl_algo"
            )
        cfg.grpo_kl_beta = 0.0
        v = getattr(ev, "blb_v3_warmstart_bias_gain", None)
        if v not in (None, ""):
            try:
                cfg.warmstart_bias_gain = float(v)
            except Exception:
                pass
        v = getattr(ev, "blb_v3_ent_coef", None)
        if v not in (None, ""):
            try:
                cfg.ppo.ent_coef = max(0.0, float(v))
            except Exception:
                pass
        v = getattr(ev, "blb_v3_ent_coef_anchor", None)
        if v not in (None, ""):
            try:
                cfg.ent_coef_anchor = max(0.0, float(v))
            except Exception:
                pass
        v = getattr(ev, "blb_v3_warmstart_baseline_bias", None)
        if v not in (None, ""):
            cfg.warmstart_baseline_bias = str(v).strip().lower() not in (
                "0", "false", "no", "off",
            )

        # rollout_size 上限不能超过 total_episodes
        v = getattr(ev, "blb_v3_warmstart_neighbor_sampling", None)
        if v not in (None, ""):
            cfg.warmstart_neighbor_sampling = str(v).strip().lower() not in (
                "0", "false", "no", "off",
            )
        v = getattr(ev, "blb_v3_guarded_radius2_enabled", None)
        if v not in (None, ""):
            cfg.guarded_radius2_enabled = str(v).strip().lower() not in (
                "0", "false", "no", "off",
            )
        v = getattr(ev, "blb_v3_static_invalid_level_mask_enabled", None)
        if v not in (None, ""):
            cfg.static_invalid_level_mask_enabled = str(v).strip().lower() not in (
                "0", "false", "no", "off",
            )
        v = getattr(ev, "blb_v3_fast_reward_mode_enabled", None)
        if v not in (None, ""):
            cfg.fast_reward_mode_enabled = str(v).strip().lower() in (
                "1", "true", "yes", "on",
            )
        v = getattr(ev, "blb_v3_guarded_radius2_episode_fraction", None)
        if v not in (None, ""):
            try:
                cfg.guarded_radius2_episode_fraction = float(v)
            except Exception:
                pass
        v = getattr(ev, "blb_v3_promotion_margin_window", None)
        if v not in (None, ""):
            try:
                cfg.promotion_margin_window = float(v)
            except Exception:
                pass
        for cfg_field, attr_name in (
                ("stage2_stability_multiplier", "stage2_stability_multiplier"),
                ("online_constraint_probability", "blb_v3_online_constraint_probability"),
                ("promotion_constraint_probability", "blb_v3_promotion_constraint_probability"),
                ("final_constraint_probability", "blb_v3_final_constraint_probability"),
        ):
            value = getattr(ev, attr_name, None)
            if value not in (None, ""):
                setattr(cfg, cfg_field, float(value))
        v = getattr(ev, "blb_v3_disable_warmstart_on_resume", None)
        if v not in (None, ""):
            cfg.disable_warmstart_on_resume = str(v).strip().lower() in (
                "1", "true", "yes", "on",
            )
        v = getattr(ev, "blb_v3_action_mask_enabled", None)
        if v not in (None, ""):
            cfg.action_mask_enabled = str(v).strip().lower() in (
                "1", "true", "yes", "on",
            )
        v = getattr(ev, "blb_v3_action_mask_mode", None)
        if v not in (None, ""):
            cfg.action_mask_mode = str(v).strip()
        v = getattr(ev, "blb_v3_action_mask_file", None)
        if v not in (None, ""):
            cfg.action_mask_file = str(v).strip()
        v = getattr(ev, "blb_v3_action_mask_source", None)
        if v not in (None, ""):
            cfg.action_mask_source = str(v).strip()
        v = getattr(ev, "blb_v3_action_mask_baseline_logit_bonus", None)
        if v not in (None, ""):
            try:
                cfg.action_mask_baseline_logit_bonus = float(v)
            except Exception:
                pass
        if str(cfg.action_mask_mode or "").strip().lower() not in ("", "none", "off", "disabled"):
            cfg.action_mask_enabled = True

        # Per-block sequential RL toggle (default True since 2026-05-15).
        v = getattr(ev, "blb_v3_sequential_rl", None)
        if v not in (None, ""):
            cfg.sequential_rl = str(v).strip().lower() not in (
                "0", "false", "no", "off",
            )
        # 4-sub-stage mode (opt-in 2026-05-27). When set, takes priority over
        # sequential_rl in BLBStage2RLRunner.run dispatch.
        v = getattr(ev, "blb_v3_substage_mode", None)
        if v not in (None, ""):
            cfg.substage_mode = str(v).strip().lower() in (
                "1", "true", "yes", "on",
            )
        v = getattr(ev, "blb_v3_fusion_count_action", None)
        if v not in (None, ""):
            cfg.fusion_count_action = str(v).strip().lower() in (
                "1", "true", "yes", "on",
            )
        from .layerwise_runner import apply_public_stage2_decision_config
        apply_public_stage2_decision_config(ev, cfg)
        v = getattr(ev, "blb_v3_fusion_neighbor_curriculum", None)
        if v not in (None, ""):
            cfg.fusion_neighbor_curriculum_enabled = str(v).strip().lower() in (
                "1", "true", "yes", "on",
            )
        v = getattr(ev, "blb_v3_fusion_probe_interval", None)
        if v not in (None, ""):
            try:
                cfg.fusion_probe_interval = int(v)
            except Exception:
                pass
        v = getattr(ev, "blb_v3_fusion_exploration_epsilon", None)
        if v not in (None, ""):
            try:
                cfg.fusion_exploration_epsilon = float(v)
            except Exception:
                pass
        if bool(getattr(cfg, "fusion_count_action", False)) and bool(getattr(cfg, "substage_mode", False)):
            raise ValueError(
                "blb_v3_fusion_count_action and blb_v3_substage_mode are mutually exclusive"
            )
        v = getattr(ev, "blb_v3_substage_block_order", None)
        if v not in (None, ""):
            try:
                cfg.substage_block_order = parse_int_list_text(
                    str(v), allow_semicolon=False,
                )
            except Exception:
                pass
        v = getattr(ev, "blb_v3_substage_frozen_blocks", None)
        if v not in (None, ""):
            try:
                cfg.substage_frozen_blocks = parse_int_list_text(
                    str(v), allow_semicolon=False,
                )
            except Exception:
                pass
        v = getattr(ev, "blb_v3_substage_episodes_each", None)
        if v not in (None, ""):
            try:
                cfg.substage_episodes_each = int(v)
            except Exception:
                pass
        v = getattr(ev, "blb_v3_substage_promotion_top_k", None)
        if v not in (None, ""):
            try:
                cfg.substage_promotion_top_k = int(v)
            except Exception:
                pass
        v = getattr(ev, "blb_v3_substage_promotion_trials", None)
        if v not in (None, ""):
            try:
                cfg.substage_promotion_trials = int(v)
            except Exception:
                pass
        # OSR pre-prune flags (2026-05-27 opt-in).
        v = getattr(ev, "blb_v3_osr_results_path", None)
        if v not in (None, ""):
            cfg.osr_results_path = str(v)
        v = getattr(ev, "blb_v3_osr_scan_only", None)
        if v not in (None, ""):
            cfg.osr_scan_only = str(v).strip().lower() in ("1", "true", "yes", "on")
        v = getattr(ev, "blb_v3_osr_num_combo_samples", None)
        if v not in (None, ""):
            try:
                cfg.osr_num_combo_samples = max(0, int(v))
            except Exception:
                pass
        v = getattr(ev, "blb_v3_osr_allow_fingerprint_mismatch", None)
        if v not in (None, ""):
            cfg.osr_allow_fingerprint_mismatch = str(v).strip().lower() in (
                "1", "true", "yes", "on",
            )
        for cfg_field, attr_name, caster in (
                ("sequential_invalid_penalty", "blb_v3_sequential_invalid_penalty", float),
                ("sequential_cost_shaping_coeff", "blb_v3_sequential_cost_shaping_coeff", float),
                ("sequential_fusion_shaping_coeff", "blb_v3_sequential_fusion_shaping_coeff", float),
        ):
            v = getattr(ev, attr_name, None)
            if v not in (None, ""):
                try:
                    setattr(cfg, cfg_field, caster(v))
                except Exception:
                    pass
        v = getattr(ev, "blb_v3_sequential_early_terminate_on_invalid", None)
        if v not in (None, ""):
            cfg.sequential_early_terminate_on_invalid = str(v).strip().lower() in (
                "1", "true", "yes", "on",
            )

        # Multi-seed support: when --blb-v3-seed is provided, override the
        # default training seed (BLBStage2TrainConfig.seed=42). Used by
        # tools/run_multi_seed.sh to sweep N seeds for significance testing.
        v = getattr(ev, "blb_v3_seed", None)
        if v not in (None, ""):
            try:
                cfg.seed = int(v)
            except Exception:
                pass

        if int(cfg.total_episodes) == 0:
            cfg.rollout_size = max(1, int(cfg.rollout_size))
        else:
            cfg.rollout_size = max(
                1, min(int(cfg.rollout_size), int(cfg.total_episodes))
            )
        if cfg.warmstart_anchor_episodes is None:
            cfg.warmstart_anchor_episodes = 0
        else:
            cfg.warmstart_anchor_episodes = max(
                0,
                min(int(cfg.warmstart_anchor_episodes), int(cfg.total_episodes)),
            )
        if cfg.warmstart_neighbor_ramp_episodes is None:
            cfg.warmstart_neighbor_ramp_episodes = 0
        else:
            cfg.warmstart_neighbor_ramp_episodes = max(
                0,
                min(int(cfg.warmstart_neighbor_ramp_episodes), int(cfg.total_episodes)),
            )
        cfg.warmstart_neighbor_max_mutations = max(
            1, min(int(cfg.warmstart_neighbor_max_mutations), 64),
        )
        cfg.warmstart_neighbor_max_radius = max(
            1, min(int(cfg.warmstart_neighbor_max_radius), 8),
        )
        cfg.guarded_radius2_min_episode = max(0, int(cfg.guarded_radius2_min_episode))
        cfg.guarded_radius2_stall_window = max(1, int(cfg.guarded_radius2_stall_window))
        cfg.guarded_radius2_health_window = max(1, int(cfg.guarded_radius2_health_window))
        cfg.guarded_radius2_max_mutations = max(
            1, min(int(cfg.guarded_radius2_max_mutations), 16),
        )
        cfg.guarded_radius2_episode_fraction = float(
            np.clip(float(cfg.guarded_radius2_episode_fraction), 0.0, 1.0)
        )
        cfg.guarded_radius2_cooldown_episodes = max(
            0, int(cfg.guarded_radius2_cooldown_episodes),
        )
        cfg.guarded_radius2_min_radius1_successes = max(
            1, int(cfg.guarded_radius2_min_radius1_successes),
        )
        cfg.online_num_trials_per_step = max(1, int(cfg.online_num_trials_per_step))
        cfg.terminal_eval_batch_size = max(1, int(cfg.terminal_eval_batch_size))
        cfg.promotion_validation_trials = max(1, int(cfg.promotion_validation_trials))
        cfg.final_selection_top_n = max(1, int(cfg.final_selection_top_n))
        cfg.final_selection_validation_trials = max(1, int(cfg.final_selection_validation_trials))
        cfg.promotion_margin_window = max(0.0, float(cfg.promotion_margin_window))
        cfg.ent_coef_ramp_episodes = max(
            0, min(int(cfg.ent_coef_ramp_episodes), int(cfg.total_episodes))
        )
        cfg.validate_decision_granularity()
        cfg.validate_reward_design()
        cfg.validate_robust_constraint_config()
        from .layerwise_runner import validate_stage2_episode_limit_mode
        validate_stage2_episode_limit_mode(
            int(cfg.total_episodes),
            fusion_count_action=bool(cfg.fusion_count_action),
            decision_granularity=cfg.decision_granularity,
            reward_design=cfg.reward_design,
            sequential_rl=bool(cfg.sequential_rl),
            substage_mode=bool(cfg.substage_mode),
            stage2_rl_variant="blb_v3",
        )
        return cfg

    def _build_probe_batches(
            self,
            ev,
            train_cfg: BLBStage2TrainConfig,
            ) -> List[ProbeBatch]:
        """构造 RL 评估子集：使用 evaluator 已有的 stability probe。"""
        device = ev.device
        split_name = ev.get_reward_reference_split_name()
        probe_size = int(getattr(ev, "stage2_probe_size", 256))

        try:
            probe_subset, probe_subset_mm = ev._get_stability_probe(
                split_name, probe_size, probe_seed=int(train_cfg.seed),
            )
        except Exception:
            probe_subset = None

        if probe_subset is None:
            ds = ev.dataset_splits.get(split_name) or ev.dataset_splits.get("train")
            if ds is None:
                return []
            probe_subset = ds

        # 构造 dataloader
        from torch.utils.data import DataLoader
        loader = DataLoader(
            probe_subset,
            batch_size=int(ev.batch_size),
            shuffle=False,
            collate_fn=ev.data_collator,
            pin_memory=torch.cuda.is_available(),
        )
        max_count = _effective_probe_batch_count(ev, train_cfg)
        out: List[ProbeBatch] = []
        for batch in loader:
            out.append(ProbeBatch.from_batch(batch, torch.device(device)))
            if len(out) >= max_count:
                break
        return out

    def _build_rescale_bridge(
            self,
            train_cfg: BLBStage2TrainConfig,
            log,
            ) -> RescaleOptimizerBridge:
        from rescale_optimizer_bridge import InProcessInvoker

        root = str(
            train_cfg.inproc_rescale_optimizer_root
            or os.path.join(_resolve_repo_root(), "Rescale_optimizer")
        )
        profile = str(train_cfg.inproc_profile or train_cfg.profile)
        log(f"  * Rescale_optimizer 根目录：{root}")
        log("  * Rescale optimizer 模式：in_process_real")

        try:
            if train_cfg.inproc_configs:
                baseline = train_cfg.inproc_baseline_archive
                if not baseline:
                    raise ValueError(
                        "inproc_configs requires inproc_baseline_archive for real Rescale_optimizer"
                    )
                invoker = InProcessInvoker(
                    configs=dict(train_cfg.inproc_configs),
                    baseline_archive=str(baseline),
                    rescale_optimizer_root=root,
                )
            else:
                invoker = InProcessInvoker.from_profile(
                    rescale_optimizer_root=root,
                    profile=profile,
                    baseline_archive=train_cfg.inproc_baseline_archive,
                )
            if not getattr(invoker, "baselines", {}):
                raise ValueError(
                    f"no Rescale_optimizer baselines loaded for profile={profile!r}"
                )
        except Exception as exc:
            raise RuntimeError(
                "BLB Stage-2 RL requires the real Rescale_optimizer in-process path. "
                f"Failed to initialize from root={root!r}, profile={profile!r}: {exc}"
            ) from exc

        return RescaleOptimizerBridge(invoker=invoker)

    @staticmethod
    def _dominant_degree(degrees, default=4) -> int:
        arr = np.asarray(degrees, dtype=int).reshape(-1)
        if arr.size == 0:
            return int(default)
        vals, counts = np.unique(arr, return_counts=True)
        return int(vals[np.argmax(counts)])

    # ------------------------------------------------------------------
    # baseline metrics（精度 / 稳定性）估计
    # ------------------------------------------------------------------
    def _estimate_baseline_metrics(self, env: BLBStage2Env):
        """在不装 BLB 的前提下，跑 K trials 评估子集，得到 baseline 精度 / std。

        这段会**短暂污染** RNG state（不带 BLB，所以模型 forward 是确定性的，但
        我们仍按 K trials 跑以保持 EpisodeMetrics 的语义）。
        """
        return env._eval_on_probe(env.env_cfg.num_trials_per_step)

    # ------------------------------------------------------------------
    # checkpoint
    # ------------------------------------------------------------------
    @staticmethod
    def _torch_load_checkpoint(path: str, *, map_location):
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)

    @staticmethod
    def _rng_state_dict() -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "torch_cpu": torch.get_rng_state(),
            "numpy": np.random.get_state(),
        }
        if torch.cuda.is_available():
            try:
                state["torch_cuda"] = torch.cuda.get_rng_state_all()
            except Exception:
                pass
        return state

    @staticmethod
    def _restore_rng_state(state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            return
        try:
            if state.get("torch_cpu") is not None:
                torch.set_rng_state(state["torch_cpu"])
        except Exception:
            pass
        try:
            if state.get("numpy") is not None:
                np.random.set_state(state["numpy"])
        except Exception:
            pass
        try:
            if torch.cuda.is_available() and state.get("torch_cuda") is not None:
                torch.cuda.set_rng_state_all(state["torch_cuda"])
        except Exception:
            pass

    @staticmethod
    def _train_cfg_to_dict(train_cfg: BLBStage2TrainConfig) -> Dict[str, Any]:
        try:
            return asdict(train_cfg)
        except Exception:
            return {
                "total_episodes": int(getattr(train_cfg, "total_episodes", 0)),
                "rollout_size": int(getattr(train_cfg, "rollout_size", 0)),
                "save_interval": int(getattr(train_cfg, "save_interval", 0)),
                "eval_interval": int(getattr(train_cfg, "eval_interval", 0)),
                "profile": str(getattr(train_cfg, "profile", "")),
                "rescale_optimizer": "in_process_real",
                "rescale_optimizer_root": str(getattr(train_cfg, "inproc_rescale_optimizer_root", "")),
            }

    @staticmethod
    def _mark_stage2_stopped(ev, *, completed_episodes: int, total_episodes: int) -> None:
        run_output_dir = getattr(ev, "run_output_dir", "")
        if not run_output_dir:
            return
        try:
            from layer_importance_evaluator import update_persistent_metadata_stage
            update_persistent_metadata_stage(
                run_output_dir,
                "stage2_search",
                "in_progress",
                extra_fields={
                    "completed_episodes": int(completed_episodes),
                    "total_episodes": int(total_episodes),
                    "stopped_by": "graceful_stop",
                    "rl_variant": "blb_v3",
                },
            )
        except Exception:
            pass

    def _save_checkpoint(
            self,
            ev,
            policy: BLBStage2Policy,
            optimizer: torch.optim.Optimizer,
            episode: int,
            best_reward: float,
            best_action: Optional[np.ndarray],
            label: str = "live",
            best_breakdown: Optional[Dict[str, Any]] = None,
            best_decoded_pickle: Optional[bytes] = None,
            episode_returns: Optional[Sequence[float]] = None,
            update_count: int = 0,
            fixed_gelu=None,
            fixed_softmax=None,
            train_cfg: Optional[BLBStage2TrainConfig] = None,
            best_reward_curve: Optional[Sequence[float]] = None,
            ppo_loss_curve: Optional[Sequence[float]] = None,
            ) -> str:
        filename = (
            BLB_STAGE2_FINAL_CHECKPOINT_FILENAME
            if str(label) == "final"
            else BLB_STAGE2_LIVE_CHECKPOINT_FILENAME
        )
        path = os.path.join(
            ev.noise_stage_progress_dir, filename,
        )
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            payload = {
                "policy": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "episode": int(episode),
                "completed_episodes": int(episode),
                "ppo_update_count": int(update_count),
                "episode_returns": [float(x) for x in (episode_returns or [])],
                # Persist curve buffers so resume keeps full-history plots
                # (Bug #4 fix: previously only ``episode_returns`` was saved
                # so ``best_reward_curve`` and ``ppo_loss_curve`` got truncated
                # to the latest session after every resume).
                "best_reward_curve": [float(x) for x in (best_reward_curve or [])],
                "ppo_loss_curve": [float(x) for x in (ppo_loss_curve or [])],
                "best_reward": float(best_reward),
                "best_action": (
                    best_action.tolist() if best_action is not None else None
                ),
                "best_breakdown": dict(best_breakdown or {}),
                "best_decoded_pickle": best_decoded_pickle,
                "fixed_gelu": (
                    np.asarray(fixed_gelu, dtype=int).tolist()
                    if fixed_gelu is not None else None
                ),
                "fixed_softmax": (
                    np.asarray(fixed_softmax, dtype=int).tolist()
                    if fixed_softmax is not None else None
                ),
                "train_cfg": (
                    self._train_cfg_to_dict(train_cfg) if train_cfg is not None else {}
                ),
                "profile": str(getattr(train_cfg, "profile", "")) if train_cfg is not None else "",
                "rng_state": self._rng_state_dict(),
                "rl_variant": "blb_v3",
            }
            tmp_path = path + ".tmp"
            try:
                torch.save(payload, tmp_path)
                os.replace(tmp_path, path)
            finally:
                if os.path.isfile(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
        except Exception as exc:
            ev.log(f"  [save_checkpoint][警告] 保存 {path} 失败: {exc}")
        return path

    @staticmethod
    def _make_log_safe(log_fn):
        """包装一层 log 函数，把控制台不能编码的字符自动 fallback 成 ASCII。

        Why: ``LayerImportanceEvaluator.log`` 同时 print 到 stdout 和写文件
        (encoding=utf-8)。Windows 默认 GBK 控制台对非 GBK Unicode 字符（如
        ▸ U+25B8）会抛 UnicodeEncodeError。我们把控制台不能编码的字符替换成
        '?' 之后再调原 log_fn，这样既不影响日志文件内容（utf-8 写入正常），
        又避免 stdout 编码错误。
        """
        import sys

        def safe_log(message):
            try:
                return log_fn(message)
            except UnicodeEncodeError:
                # 双保险：用 stdout 实际编码替换不能编码的字符
                enc = getattr(sys.stdout, "encoding", "utf-8") or "utf-8"
                try:
                    safe = str(message).encode(enc, errors="replace").decode(enc)
                    return log_fn(safe)
                except Exception:
                    return log_fn(str(message).encode("ascii", errors="replace").decode("ascii"))

        return safe_log

    @staticmethod
    def _format_action_diff(
            prev_action_vec: np.ndarray,
            curr_action_vec: np.ndarray,
            slot_labels: Sequence[str],
            *,
            limit: int = 5,
            ) -> str:
        """``slot_label idx_old→idx_new`` for the slots that changed.

        Slot label uses the compact ``L{i}.B{n}.{kind}[.{short}]`` scheme so
        the user can locate every change inside the BLB flow without an extra
        lookup table.
        """
        prev = np.asarray(prev_action_vec, dtype=int).reshape(-1)
        curr = np.asarray(curr_action_vec, dtype=int).reshape(-1)
        if prev.size != curr.size:
            return f"<size mismatch prev={prev.size} curr={curr.size}>"
        diffs = np.where(prev != curr)[0]
        if diffs.size == 0:
            return "(no change)"
        parts: List[str] = []
        for idx in diffs[: max(1, int(limit))]:
            label = slot_labels[int(idx)] if int(idx) < len(slot_labels) else f"#{int(idx)}"
            parts.append(f"{label} {int(prev[idx])}→{int(curr[idx])}")
        if diffs.size > int(limit):
            parts.append(f"... (+{int(diffs.size - limit)} more)")
        return "; ".join(parts)

    @staticmethod
    def _aggregate_entropy_by_kind(
            per_dim_entropy: np.ndarray,
            kind_by_index: Sequence[str],
            ) -> Dict[str, float]:
        """Group per-dim entropy by slot kind (F/W/M/S/R/K) → mean entropy.

        Total entropy hides per-kind collapse; surfacing it surfaces whether
        e.g. all R-slots collapsed early (CLAUDE.md "warmstart toward all-max
        baseline" + "per-slot entropy logging").
        """
        ent = np.asarray(per_dim_entropy, dtype=float).reshape(-1)
        out: Dict[str, float] = {}
        for kind in ("F", "W", "M", "S", "R", "K"):
            mask = np.array([k == kind for k in kind_by_index], dtype=bool)
            if mask.size != ent.size:
                continue
            if mask.any():
                out[kind] = float(ent[mask].mean())
        return out

    @staticmethod
    def _breakdown_to_dict(breakdown) -> Dict[str, Any]:
        if breakdown is None:
            return {}
        return {
            "reward": float(breakdown.reward),
            "priority": int(breakdown.priority),
            "invalid": bool(breakdown.invalid),
            "r_bits": float(breakdown.r_bits),
            "r_fusion": float(breakdown.r_fusion),
            "r_k": float(breakdown.r_k),
            "bits_drop": float(breakdown.bits_drop),
            "k_drop": float(breakdown.k_drop),
            "fusion_count": float(breakdown.fusion_count),
            "acc_violation": float(breakdown.acc_violation),
            "stab_violation": float(breakdown.stab_violation),
        }

    @staticmethod
    def _metrics_to_dict(metrics) -> Dict[str, float]:
        if metrics is None:
            return {}
        out: Dict[str, float] = {}
        for key in (
                "loss_mean", "loss_std", "metric1_mean", "metric2_mean",
                "loss_max", "metric1_min", "metric2_min"):
            try:
                out[key] = float(getattr(metrics, key))
            except Exception:
                pass
        return out

    # ------------------------------------------------------------------
    # 训练日志
    # ------------------------------------------------------------------
    @staticmethod
    def _log_train_iter(
            log,
            episode: int,
            total_episodes: int,
            recent_returns: Sequence[float],
            metrics: Mapping[str, Any],
            best_reward: float,
            ) -> None:
        rr = np.asarray(list(recent_returns), dtype=float) if recent_returns else None
        rr_mean = float(rr.mean()) if rr is not None and rr.size > 0 else 0.0
        rr_max = float(rr.max()) if rr is not None and rr.size > 0 else 0.0
        log(
            _format_blb_train_iter_log(
                episode=episode,
                total_episodes=total_episodes,
                return_mean=rr_mean,
                return_max=rr_max,
                best_reward=best_reward,
                policy_loss=float(metrics.get('policy_loss', 0.0)),
                value_loss=float(metrics.get('value_loss', 0.0)),
                entropy=float(metrics.get('entropy', 0.0)),
                clip_fraction=float(metrics.get('clip_fraction', 0.0)),
            )
        )
