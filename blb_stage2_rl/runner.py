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
import os
import pickle
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from rescale_optimizer_bridge import RescaleOptimizerBridge
from .action_space import (
    K_LEVELS,
    MaxSFsTable,
    action_dims_for_config,
    action_vector_to_cfgs,
    avg_truncation_k_in_action,
    layer_dims,
    load_max_sfs,
    make_all_max_action_vector,
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
)
from .persistence import (
    BLBStatusBoard,
    dump_crash_report,
    write_blb_final_report,
    write_training_curves,
)


BLB_STAGE2_LIVE_CHECKPOINT_FILENAME = "blb_stage2_rl_checkpoint_live.pt"
BLB_STAGE2_FINAL_CHECKPOINT_FILENAME = "blb_stage2_rl_checkpoint_final.pt"
BLB_STAGE2_BEST_CFG_FILENAME = "blb_stage2_best_cfg.pkl"

# ---------------------------------------------------------------------------
# BLB Stage 2 RL 专属持久化根目录（与旧 stage 2 RL 完全分开）
# ---------------------------------------------------------------------------
# 旧版 stage 2 RL (noise_rl_module_v2) 把 checkpoint/曲线/状态都丢在 evaluator
# 给的 ``noise_stage_progress_dir``（默认 ``rl_results/persistent/...``）下。
# BLB Stage 2 RL 是"最终版本"——为了让用户能清晰区分两套产物，新版的所有
# 持久化文件都搬到项目根下的 ``Parting Chapter/`` 目录里。
#
# 当 evaluator 提供了 ``run_output_dir`` 时，最终路径形如
#   <repo_root>/Parting Chapter/<basename(run_output_dir)>/blb_stage2/
# 否则 fallback 到 ``<repo_root>/Parting Chapter/blb_stage2_default_run/``。
#
# Why "Parting Chapter"：这是新的 BLB Stage-2 持久化章节名；旧目录只作为
# 历史归档保留，不再写入。
BLB_PARTING_CHAPTER_DIRNAME = "Parting Chapter"


def _resolve_repo_root() -> str:
    """回到项目根目录。``runner.py`` 位于 ``<root>/blb_stage2_rl/``，向上一级即可。"""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(here)


def resolve_blb_persistence_dir(evaluator) -> str:
    """计算 BLB Stage 2 RL 的持久化目录（覆盖 ``ev.noise_stage_progress_dir``）。

    输出形如 ``<repo_root>/Parting Chapter/<run_basename>/blb_stage2/progress``。
    若 ``evaluator.run_output_dir`` 为空，使用 ``blb_stage2_default_run``。
    """
    repo_root = _resolve_repo_root()
    run_dir = str(getattr(evaluator, "run_output_dir", "") or "").strip()
    if run_dir:
        run_basename = os.path.basename(os.path.normpath(run_dir)) or "blb_stage2_default_run"
    else:
        run_basename = "blb_stage2_default_run"
    out = os.path.join(repo_root, BLB_PARTING_CHAPTER_DIRNAME, run_basename, "blb_stage2", "progress")
    os.makedirs(out, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# CLI 友好的 RL 训练超参（给 evaluator / rl_tune 透传）
# ---------------------------------------------------------------------------
@dataclass
class BLBStage2TrainConfig:
    """``BLBStage2RLRunner`` 的可配置训练参数。"""
    total_episodes: int = 2000              # 默认 2000 episodes
    rollout_size: int = 32                  # 每多少 episode 触发一次 PPO update
    seed: int = 42
    eval_interval: int = 100                # 多少 episode 跑一次 deterministic eval
    save_interval: int = 200
    profile: str = "default"
    # spec §6.4 / §3.1
    acc_threshold: float = 0.0              # baseline 精度往下浮 1pp 后用此值
    stab_threshold: float = float("inf")
    # PPO
    ppo: PPOConfig = field(default_factory=PPOConfig)
    # 环境
    num_trials_per_step: int = 3
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
        # ---------- 0.1) 切换到 BLB Stage 2 RL 专属持久化根目录 ----------
        # 把 evaluator 的 noise_stage_progress_dir 覆盖成 "Parting Chapter/.../blb_stage2/progress"。
        # 旧 stage 2 RL 的目录 (rl_results/persistent/...) 不动 —— 这两套互不影响。
        legacy_progress_dir = str(getattr(ev, "noise_stage_progress_dir", "") or "")
        blb_progress_dir = resolve_blb_persistence_dir(ev)
        try:
            ev.noise_stage_progress_dir = blb_progress_dir
        except Exception:
            pass
        log("\n" + "=" * 80)
        log("阶段 5 · 加强版 BLB Stage 2 强化学习（BLB Stage 2 RL · v3）")
        log("=" * 80)
        log(f"  {bullet} 固定 GELU/Softmax 来源：{fixed_source}    标签：{fixed_label}")
        log(f"  {bullet} GELU 离散阶数向量:   {np.asarray(fixed_gelu, dtype=int).tolist()}")
        log(f"  {bullet} Softmax 离散阶数向量: {np.asarray(fixed_softmax, dtype=int).tolist()}")
        log(f"  {bullet} Profile（数据集）= {train_cfg.profile!r}    "
            f"Episode 总数 = {train_cfg.total_episodes}    "
            f"PPO 更新间隔（rollout_size） = {train_cfg.rollout_size}")
        log(f"  {bullet} BLB 持久化目录 = {blb_progress_dir}")
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
        log(f"  {bullet} 评估子集 batch 数 = {len(probe_batches)}")

        # ---------- 3) 准备 RescaleOptimizer 桥 ----------
        rescale_bridge = self._build_rescale_bridge(train_cfg, log=log)

        # ---------- 4) 准备 max_sfs 表 + Env ----------
        max_sfs = load_max_sfs(train_cfg.profile)
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

        # ---------- 5) baseline + reward 权重校准 ----------
        status.set_phase("校准 baseline cost / reward 权重")
        baseline = estimate_baseline_cost_stats(
            env, sample_count=int(train_cfg.calibrate_baseline_samples),
        )
        env.baseline = baseline
        weights = calibrate_weights_from_baseline(baseline)
        env.reward_weights = weights
        status.set_baseline({
            "total_bits_sum": int(baseline.total_bits_sum),
            "total_fusion_count": int(baseline.total_fusion_count),
            "avg_k": float(baseline.avg_k),
            "typical_bits_drop": float(baseline.typical_bits_drop),
            "typical_fusion_count": float(baseline.typical_fusion_count),
            "typical_k_drop": float(baseline.typical_k_drop),
        })
        status.set_extra("reward_weights", {
            "w_bits": float(weights.w_bits),
            "w_fusion": float(weights.w_fusion),
            "w_k": float(weights.w_k),
        })
        log(
            f"  {bullet} Baseline cost: total_bits_sum={baseline.total_bits_sum}, "
            f"total_fusion_count={baseline.total_fusion_count}, avg_k={baseline.avg_k:.2f}"
        )
        log(
            f"  {bullet} Reward weights: w_bits={weights.w_bits:.6g}, "
            f"w_fusion={weights.w_fusion:.4g}, w_k={weights.w_k:.4g}"
        )

        # 估计 baseline 精度 + 稳定性，用于硬阈值校准
        baseline_metrics = self._estimate_baseline_metrics(env)
        baseline.loss_mean = float(baseline_metrics.loss_mean)
        baseline.loss_std = float(baseline_metrics.loss_std)
        baseline.metric1_mean = float(baseline_metrics.metric1_mean)
        baseline.metric2_mean = float(baseline_metrics.metric2_mean)

        if not np.isfinite(env.acc_threshold) or env.acc_threshold <= 0.0:
            env.acc_threshold = max(0.0, float(baseline.metric1_mean) - 0.01)   # 1pp
        if not np.isfinite(env.stab_threshold):
            env.stab_threshold = float(baseline.loss_std) * 1.5 + 1e-3
        log(
            f"  {bullet} Baseline metrics: loss_mean={baseline.loss_mean:.4f}, "
            f"loss_std={baseline.loss_std:.4f}, m1={baseline.metric1_mean:.4f}, "
            f"m2={baseline.metric2_mean:.4f}"
        )
        log(
            f"  {bullet} 硬约束阈值: acc_threshold={env.acc_threshold:.4f}, "
            f"stab_threshold={env.stab_threshold:.4f}"
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
        optimizer = torch.optim.Adam(policy.parameters(), lr=float(train_cfg.ppo.lr))

        # ---------- 7) Resume（可选） ----------
        start_episode = 0
        update_count = 0
        best_reward = -float("inf")
        best_action_vec: Optional[np.ndarray] = None
        best_breakdown_dict: Optional[Dict[str, Any]] = None
        best_decoded_pickle: Optional[bytes] = None
        episode_returns: List[float] = []
        if resume_checkpoint_path and os.path.isfile(resume_checkpoint_path):
            try:
                ckpt = self._torch_load_checkpoint(resume_checkpoint_path, map_location=device)
                if isinstance(ckpt, dict):
                    policy.load_state_dict(ckpt["policy"])
                    optimizer.load_state_dict(ckpt["optimizer"])
                    start_episode = int(ckpt.get("completed_episodes", ckpt.get("episode", 0)))
                    update_count = int(ckpt.get("ppo_update_count", 0))
                    episode_returns = [float(x) for x in ckpt.get("episode_returns", [])]
                    if "best_reward" in ckpt:
                        best_reward = float(ckpt["best_reward"])
                    if ckpt.get("best_action") is not None:
                        best_action_vec = np.asarray(ckpt["best_action"], dtype=int)
                    best_breakdown_dict = ckpt.get("best_breakdown")
                    best_decoded_pickle = ckpt.get("best_decoded_pickle")
                    rng_state = ckpt.get("rng_state") or {}
                    self._restore_rng_state(rng_state)
                    log(f"  {bullet} Resumed from {resume_checkpoint_path} (episode={start_episode})")
            except Exception as exc:
                log(f"  [resume][警告] 读取 checkpoint 失败：{exc}")

        # ---------- 8) 训练循环 ----------
        status.set_phase(f"训练中（PPO 单步 episode，共 {train_cfg.total_episodes} 回合）")
        log("\n训练开始（PPO 单步 episode）...")
        buffer = RolloutBuffer()
        env.reset(seed=int(train_cfg.seed))
        stop_flag_path = None
        graceful_stop_logged = False
        # 训练曲线累积（与 episode_returns 配套；ppo_loss_curve 每次 PPO update 追加）
        best_reward_curve: List[float] = []
        ppo_loss_curve: List[float] = []
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

        try:
            for ep in range(start_episode, int(train_cfg.total_episodes)):
                obs = env.reset()

                obs_t = torch.from_numpy(obs).float().to(device).unsqueeze(0)
                with torch.no_grad():
                    action_t, log_prob_t, value_t = policy.sample_action(obs_t, deterministic=False)
                action_vec = action_t.squeeze(0).cpu().numpy().astype(np.int64)
                log_prob = float(log_prob_t.item())
                value = float(value_t.item())

                obs_next, reward, done, info = env.step(action_vec)

                buffer.add(state=obs, action=action_vec, log_prob=log_prob,
                           reward=float(reward), value=value)
                episode_returns.append(float(reward))

                # 跟踪 best
                if float(reward) > best_reward:
                    best_reward = float(reward)
                    best_action_vec = action_vec.copy()
                    breakdown = info.get("reward_breakdown")
                    best_breakdown_dict = self._breakdown_to_dict(breakdown) if breakdown else None
                    status.set_best(
                        best_reward=best_reward,
                        best_action_vec=best_action_vec,
                        best_breakdown=best_breakdown_dict,
                        best_episode=int(ep + 1),
                    )
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
                status.update_after_episode(int(ep + 1), float(reward), breakdown=best_breakdown_dict)

                did_update = False
                # PPO update
                if len(buffer) >= int(train_cfg.rollout_size):
                    metrics = ppo_update(policy, optimizer, buffer, train_cfg.ppo, device)
                    buffer.clear()
                    update_count += 1
                    did_update = True
                    try:
                        ppo_loss_curve.append(float(metrics.get("policy_loss", 0.0)))
                    except Exception:
                        pass
                    status.update_after_ppo_update(int(update_count), metrics)
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
                        )
                        if consume_stop_flag_file is not None:
                            consume_stop_flag_file(stop_flag_path)
                        self._mark_stage2_stopped(ev, completed_episodes=ep + 1,
                                                  total_episodes=train_cfg.total_episodes)
                        status.mark_stopped(reason="用户触发优雅停止", completed_episodes=int(ep + 1))
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
            metrics = ppo_update(policy, optimizer, buffer, train_cfg.ppo, device)
            buffer.clear()

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
        )
        log(f"\n训练完成：best_reward={best_reward:.4f}")
        log(f"  {bullet} Final policy 已保存到：{final_save_path}")

        if best_action_vec is not None:
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
                    "w_bits": float(weights.w_bits),
                    "w_fusion": float(weights.w_fusion),
                    "w_k": float(weights.w_k),
                    "acc_threshold": float(env.acc_threshold),
                    "stab_threshold": float(env.stab_threshold),
                },
                episode_returns=episode_returns,
                rescale_invoker_kind="in_process_real",
                log_fn=log,
            )
            log(f"  {bullet} 最终训练报告 → {report_path}")
            status.set_extra("final_report_path", report_path)
        except Exception as exc:
            log(f"  [警告] 写最终报告失败：{exc}")
        status.set_phase("已完成")

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

        # 用 baseline 来推算 limit_loss / limit_p / limit_s（与旧版一致）
        base_loss, base_p, base_s, _ = ev.evaluate_model(
            fixed_gelu, fixed_softmax, use_train=False,
            split=ev.get_reward_reference_split_name(),
        )
        limit_dict = ev.build_constraint_limits_from_metrics(base_loss, base_p, base_s)
        limit_loss = float(limit_dict["loss"])
        limit_p = float(limit_dict["metric1"])
        limit_s = float(limit_dict["metric2"])

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
            "limit_computation_method": "baseline_tolerance_percentage",
            "limit_tolerance": float(getattr(ev, "stage2_limit_tolerance", 0.05)),
            "stability_tolerance": float(getattr(ev, "stage2_stability_tolerance", 0.05)),
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
                "blb_v3_w_bits": float(weights.w_bits),
                "blb_v3_w_fusion": float(weights.w_fusion),
                "blb_v3_w_k": float(weights.w_k),
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
            "blb_v3_profile": str(train_cfg.profile),
            "blb_v3_total_episodes": int(train_cfg.total_episodes),
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
                ("seed", "final_eval_random_seed"),
        ):
            v = getattr(ev, attr_name, None)
            if v is None:
                continue
            try:
                setattr(cfg, cfg_field, int(v))
            except Exception:
                pass

        # rollout_size 上限不能超过 total_episodes
        cfg.rollout_size = max(1, min(int(cfg.rollout_size), int(cfg.total_episodes)))
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
        max_count = max(1, int(train_cfg.probe_batch_count))
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
        log(f"  * Rescale_optimizer root = {root}")
        log("  * Rescale optimizer mode = in_process_real")

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
            torch.save({
                "policy": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "episode": int(episode),
                "completed_episodes": int(episode),
                "ppo_update_count": int(update_count),
                "episode_returns": [float(x) for x in (episode_returns or [])],
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
                "rng_state": self._rng_state_dict(),
                "rl_variant": "blb_v3",
            }, path)
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
            f"  [BLB-v3] ep={episode}/{total_episodes}    "
            f"return mean={rr_mean:+.3f} max={rr_max:+.3f}  "
            f"best={best_reward:+.3f}  "
            f"policy_loss={float(metrics.get('policy_loss', 0.0)):.4f}  "
            f"value_loss={float(metrics.get('value_loss', 0.0)):.4f}  "
            f"entropy={float(metrics.get('entropy', 0.0)):.4f}  "
            f"clip_fraction={float(metrics.get('clip_fraction', 0.0)):.4f}"
        )
