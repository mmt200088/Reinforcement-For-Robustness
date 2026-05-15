"""Sequential PPO training loop + launcher orchestration.

Two entrypoints:

  * :func:`train_sequential` -- thin standalone driver over
    :class:`BLBStage2SequentialEnv` + :class:`BLBStage2SequentialPolicy` +
    :class:`SequentialRolloutBuffer`. Useful for A/B from a notebook.

  * :func:`run_sequential_via_runner` -- called from
    :meth:`blb_stage2_rl.runner.BLBStage2RLRunner.run` when
    ``train_cfg.sequential_rl`` is True (default on 2026-05-15). Reuses the
    existing runner's setup helpers (env construction, baseline calibration,
    persistent-dir resolution) and assembles a ``noise_stage_result`` dict in
    the same legacy-compatible shape downstream consumers expect.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np
import torch

from .sequential_env import BLBStage2SequentialEnv, SequentialEnvConfig
from .sequential_policy import (
    BLBStage2SequentialPolicy,
    SequentialPolicyConfig,
    SequentialPPOConfig,
    SequentialRolloutBuffer,
    sequential_ppo_update,
    step_to_mask_and_levels,
)


@dataclass
class SequentialTrainConfig:
    total_episodes: int = 100
    update_every_n_episodes: int = 4
    log_every_n_episodes: int = 4
    seed: Optional[int] = None
    ppo: SequentialPPOConfig = field(default_factory=SequentialPPOConfig)


# ---------------------------------------------------------------------------
# v2-style console helpers (rounded box / banner / progress bar / time fmt)
# Ported from noise_rl_module_v2 so the sequential path has the same look-and-feel.
# ---------------------------------------------------------------------------

def _seq_log_major_rule(log_fn, title: str, width: int = 68) -> None:
    """Banner: ═══════ on either side of a title line."""
    bar = "═" * int(width)
    log_fn("")
    log_fn(bar)
    log_fn(title if title.startswith(" ") else f"  {title}")
    log_fn(bar)


def _seq_block_title(log_fn, title: str) -> None:
    """Block subtitle: 【…】 brackets."""
    log_fn("")
    log_fn(f"  【{title}】")


def _seq_display_width(s: str) -> int:
    """Display columns counted by east-asian-width (CJK = 2 cols, else 1).

    This is a noticeable improvement over ``len()`` for our Chinese log lines;
    v2's ``_log_rounded_box`` uses plain ``len()`` and produces a slightly
    misaligned right border. Sequential output uses this helper instead so
    boxes line up cleanly even with mixed CJK / ASCII content.
    """
    import unicodedata
    w = 0
    for ch in str(s):
        ea = unicodedata.east_asian_width(ch)
        w += 2 if ea in ("W", "F") else 1
    return w


def _seq_ljust_display(s: str, width: int) -> str:
    """Pad ``s`` with spaces so its *display width* equals ``width``."""
    pad = max(0, int(width) - _seq_display_width(s))
    return str(s) + (" " * pad)


def _seq_log_rounded_box(log_fn, lines, indent: str = "  ", min_inner_width: int = 8) -> None:
    """Rounded box around multi-line content, aware of CJK display width."""
    stripped = [str(x) for x in lines]
    w = max((_seq_display_width(s) for s in stripped), default=0)
    w = max(w, int(min_inner_width))
    bar = "─" * (w + 4)
    log_fn(f"{indent}╭{bar}╮")
    for s in stripped:
        log_fn(f"{indent}│ {_seq_ljust_display(s, w)} │")
    log_fn(f"{indent}╰{bar}╯")


def _seq_progress_bar(current: int, total: int, width: int = 30) -> str:
    ratio = min(float(current) / max(float(total), 1.0), 1.0)
    filled = int(round(ratio * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {ratio:6.1%}"


def _seq_fmt_elapsed(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


def _seq_fmt_eta_finish(eta_seconds: float) -> str:
    finish_ts = time.time() + max(float(eta_seconds), 0.0)
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(finish_ts))


# How often (in PPO updates) to print the big progress box, matching v2's
# NOISE_RL_PROGRESS_BOX_PPO_INTERVAL.
SEQ_PROGRESS_BOX_PPO_INTERVAL = 5


@dataclass
class EpisodeRecord:
    episode_idx: int
    total_reward: float
    terminal_reward: float
    per_step_reward_sum: float
    invalid_steps: int
    early_terminated: bool
    steps_taken: int
    # Enriched fields (added 2026-05-15 for richer console output)
    valid_step_count: int = 0
    total_bits_sum_over_steps: int = 0
    fusion_count_sum_over_steps: int = 0
    first_invalid_step: Optional[int] = None
    first_invalid_block: Optional[int] = None
    first_invalid_layer: Optional[int] = None
    step_infos: List[Dict[str, Any]] = field(default_factory=list)


def train_sequential(
        *,
        env: BLBStage2SequentialEnv,
        policy: BLBStage2SequentialPolicy,
        train_cfg: Optional[SequentialTrainConfig] = None,
        device: Optional[torch.device] = None,
        on_episode_end: Optional[Callable[[EpisodeRecord], None]] = None,
        on_ppo_update_end: Optional[
            Callable[[Dict[str, float], int, "EpisodeRecord"], None]
        ] = None,
        on_step_end: Optional[
            Callable[[int, int, Dict[str, Any]], None]
        ] = None,
        capture_step_infos: bool = False,
        logger: Optional[logging.Logger] = None,
        ) -> Dict[str, object]:
    """Train ``policy`` on ``env`` with sequential PPO.

    Args:
        env:                Sequential env (horizon = 59 for L=12).
        policy:             Sequential actor-critic.
        train_cfg:          Hyper-params; defaults to a conservative SequentialTrainConfig.
        device:             Where to store tensors; defaults to model's parameter device.
        on_episode_end:     Optional callback fired at the end of each episode with
                            the EpisodeRecord. Use it to save checkpoints / status
                            / rich-log per-episode summaries.
        on_ppo_update_end:  Optional callback fired after every PPO update with
                            ``(ppo_metrics_dict, completed_episode_count, last_episode_record)``.
                            Use it to print boxed PPO summary / progress bar / ETA.
        on_step_end:        Optional callback fired after every env.step with
                            ``(episode_idx, step_idx_within_episode, info_dict)``.
                            Use sparingly (firing 59 times per episode); typical
                            pattern is to log only for the first episode.
        capture_step_infos: If True, attach the per-step info dicts to the
                            EpisodeRecord. Default False to save memory.
        logger:             Optional logger; otherwise uses module-level logging.

    Returns dict with episode_rewards / ppo_metrics / final_invalid_rate.
    """
    train_cfg = train_cfg or SequentialTrainConfig()
    device = device or next(policy.parameters()).device
    log = logger or logging.getLogger(__name__)
    optimizer = torch.optim.Adam(policy.parameters(), lr=train_cfg.ppo.lr)
    buffer = SequentialRolloutBuffer()

    episode_returns: List[float] = []
    episode_records: List[EpisodeRecord] = []
    ppo_metric_history: List[Dict[str, float]] = []

    if train_cfg.seed is not None:
        torch.manual_seed(int(train_cfg.seed))
        np.random.seed(int(train_cfg.seed) % (2**32))

    for ep in range(int(train_cfg.total_episodes)):
        # Only seed the env's RNG on the very first reset; subsequent resets
        # advance the RNG to avoid identical rollouts every episode.
        seed_for_this_ep = (
            int(train_cfg.seed) + int(ep)
            if (train_cfg.seed is not None and ep == 0)
            else None
        )
        obs = env.reset(seed=seed_for_this_ep)
        per_step_sum = 0.0
        terminal_reward = 0.0
        invalid_steps = 0
        valid_step_count = 0
        total_bits_sum = 0
        fusion_count_sum = 0
        first_invalid: Optional[Dict[str, int]] = None
        early_terminated = False
        steps_taken = 0
        captured_step_infos: List[Dict[str, Any]] = []

        while True:
            spec = env.current_spec()
            slot_mask_np, levels_np = step_to_mask_and_levels(
                spec, policy.cfg.max_step_dim, policy.cfg.max_num_levels,
            )
            obs_t = torch.from_numpy(obs).float().to(device).unsqueeze(0)
            slot_mask_t = torch.from_numpy(slot_mask_np).to(device).unsqueeze(0)
            levels_t = torch.from_numpy(levels_np).to(device).unsqueeze(0)

            with torch.no_grad():
                action_t, log_prob_t, value_t = policy.sample_action(
                    obs_t, slot_mask_t, levels_t, deterministic=False,
                )
            action_np = action_t.squeeze(0).cpu().numpy().astype(np.int64)
            log_prob = float(log_prob_t.item())
            value = float(value_t.item())

            # The policy outputs max_step_dim slots; keep only those that
            # are active for this step before passing to the env.
            n_active = int(slot_mask_np.sum())
            step_action_for_env = action_np[:n_active].tolist()

            next_obs, reward, done, info = env.step(step_action_for_env)
            steps_taken += 1
            valid = bool(info.get("valid", True))
            if not valid:
                invalid_steps += 1
                if first_invalid is None:
                    first_invalid = {
                        "step": int(info.get("step", steps_taken - 1)),
                        "block_idx": int(info.get("block_idx", 0)),
                        "layer_idx": int(info.get("layer_idx", 0)),
                    }
            else:
                valid_step_count += 1
            total_bits_sum += int(info.get("total_bits", 0))
            fusion_count_sum += int(info.get("fusion_count", 0))
            if info.get("early_terminated"):
                early_terminated = True

            # Augment info with the per-step action + reward + value for
            # downstream rich loggers (kept here so callers don't have to
            # re-track state).
            enriched_info = dict(info)
            enriched_info["action"] = step_action_for_env
            enriched_info["reward"] = float(reward)
            enriched_info["value"] = value
            enriched_info["log_prob"] = log_prob

            if on_step_end is not None:
                try:
                    on_step_end(int(ep), int(steps_taken - 1), enriched_info)
                except Exception:
                    pass
            if capture_step_infos:
                captured_step_infos.append(enriched_info)

            buffer.add(
                state=obs,
                action=action_np,
                slot_mask=slot_mask_np,
                per_slot_num_levels=levels_np,
                log_prob=log_prob,
                value=value,
                reward=float(reward),
                done=bool(done),
            )
            per_step_sum += float(reward)
            if "terminal_reward" in info:
                terminal_reward = float(info["terminal_reward"])

            obs = next_obs
            if done:
                break

        episode_returns.append(per_step_sum)
        record = EpisodeRecord(
            episode_idx=int(ep),
            total_reward=float(per_step_sum),
            terminal_reward=float(terminal_reward),
            per_step_reward_sum=float(per_step_sum - terminal_reward),
            invalid_steps=int(invalid_steps),
            early_terminated=bool(early_terminated),
            steps_taken=int(steps_taken),
            valid_step_count=int(valid_step_count),
            total_bits_sum_over_steps=int(total_bits_sum),
            fusion_count_sum_over_steps=int(fusion_count_sum),
            first_invalid_step=(int(first_invalid["step"]) if first_invalid else None),
            first_invalid_block=(int(first_invalid["block_idx"]) if first_invalid else None),
            first_invalid_layer=(int(first_invalid["layer_idx"]) if first_invalid else None),
            step_infos=captured_step_infos,
        )
        episode_records.append(record)
        if on_episode_end is not None:
            on_episode_end(record)

        if (ep + 1) % int(train_cfg.update_every_n_episodes) == 0:
            metrics = sequential_ppo_update(policy, optimizer, buffer, train_cfg.ppo, device)
            ppo_metric_history.append(metrics)
            buffer.clear()
            if on_ppo_update_end is not None:
                try:
                    on_ppo_update_end(dict(metrics), int(ep + 1), record)
                except Exception:
                    pass
            if (ep + 1) % int(train_cfg.log_every_n_episodes) == 0:
                log.info(
                    "[seqRL] ep=%d return=%.3f invalid_steps=%d ppo: %s",
                    ep, record.total_reward, record.invalid_steps, metrics,
                )

    return {
        "episode_returns": episode_returns,
        "episode_records": episode_records,
        "ppo_metrics": ppo_metric_history,
        "final_invalid_rate": (
            float(np.mean([r.invalid_steps > 0 for r in episode_records[-10:]]))
            if episode_records else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# Launcher integration: drive the sequential loop from BLBStage2RLRunner
# ---------------------------------------------------------------------------

def run_sequential_via_runner(
        *,
        runner,                           # BLBStage2RLRunner (avoid circular import)
        train_cfg,                        # BLBStage2TrainConfig
        fixed_gelu,
        fixed_softmax,
        fixed_label,
        fixed_source,
        resume_checkpoint_path=None,
        ) -> Dict[str, Any]:
    """Drive the sequential RL pipeline using BLBStage2RLRunner's setup helpers.

    Reuses ``runner._build_probe_batches``, ``runner._build_rescale_bridge``,
    ``runner._estimate_baseline_metrics``, and ``resolve_blb_persistence_dir``
    so the env / model / persistence / baseline-cost story is identical to the
    legacy single-shot path. The rollout loop is replaced with
    :func:`train_sequential` and a small bookkeeping shell saves a curve / a
    final-report markdown / a checkpoint in the same persistent directory.

    Returns a noise_stage_result dict matching the keys downstream consumers
    (UnifiedFinalEvaluationModule, BLBActionFinalEvaluationModule) read from
    the single-shot path: ``blb_v3_best_action_vec``, ``blb_v3_profile``,
    ``best_noise_config`` (legacy-compat all-max), ``limit_loss`` /
    ``limit_p`` / ``limit_s``, ``baseline_tot_c``.
    """
    import pickle

    from .baseline_bootstrap import (
        load_static_skeletons_baseline,
        static_skeletons_baseline_to_action,
    )
    from .env import BLBStage2Env, BLBStage2EnvConfig
    from .reward import BaselineCostStats, RewardWeights
    from .persistence import write_training_curves, BLBStatusBoard
    from .runner import (
        _build_legacy_compatible_best_noise_config,
        _selection_float,
        resolve_blb_persistence_dir,
    )

    ev = runner.evaluator
    bullet = "*"
    log = runner._make_log_safe(ev.log)

    # ---------- 0.1) Persistent dir ----------
    legacy_progress_dir = str(getattr(ev, "noise_stage_progress_dir", "") or "")
    blb_progress_dir = resolve_blb_persistence_dir(ev)
    try:
        ev.noise_stage_progress_dir = blb_progress_dir
    except Exception:
        pass

    _seq_log_major_rule(
        log,
        "阶段 5 · 二阶段噪声强化学习（BLB v3 · per-block sequential）",
    )
    log(f"  {bullet} 模式（mode）：horizon=59 per-block sequential（自 2026-05-15 起为默认）")
    log(f"  {bullet} 固定 GELU/Softmax 来源（source）：{fixed_source}    标签（label）：{fixed_label}")
    log(f"  {bullet} GELU 离散阶数向量:   {np.asarray(fixed_gelu, dtype=int).tolist()}")
    log(f"  {bullet} Softmax 离散阶数向量: {np.asarray(fixed_softmax, dtype=int).tolist()}")
    log(
        f"  {bullet} 训练概览：profile={train_cfg.profile!r}    "
        f"total_episodes={train_cfg.total_episodes}    "
        f"PPO 更新间隔（rollout_size）= {max(1, int(train_cfg.rollout_size))}    "
        f"seed={int(train_cfg.seed)}"
    )
    log(f"  {bullet} BLB 持久化目录：{blb_progress_dir}")

    run_basename = os.path.basename(os.path.normpath(str(getattr(ev, "run_output_dir", "") or ""))) \
        or "blb_stage2_default_run"
    status = BLBStatusBoard(
        blb_progress_dir,
        total_episodes=int(train_cfg.total_episodes),
        profile=str(train_cfg.profile),
        run_basename=run_basename,
        extra_meta={
            "fixed_label": str(fixed_label),
            "fixed_source": str(fixed_source),
            "rl_mode": "sequential_per_block",
            "rescale_optimizer": "in_process_real",
            "rescale_optimizer_root": str(train_cfg.inproc_rescale_optimizer_root),
        },
        log_fn=log,
    )
    status.set_phase("装载 stage1 GELU/Softmax 多项式近似")

    # ---------- 1) apply stage1 polynomial degrees ----------
    fixed_gelu = np.asarray(fixed_gelu, dtype=int)
    fixed_softmax = np.asarray(fixed_softmax, dtype=int)
    ev.apply_configuration(fixed_gelu, fixed_softmax)
    try:
        ev.reversible_handler.restore_layer_input_noise(
            layer_indices=list(range(ev.total_layers)),
        )
    except Exception:
        pass

    # ---------- 2) probe + bridge + baseline ----------
    probe_batches = runner._build_probe_batches(ev, train_cfg)
    train_cfg.probe_batch_count = max(1, int(len(probe_batches) or train_cfg.probe_batch_count))
    log(f"  {bullet} 评估子集：batch 数 = {len(probe_batches)}")

    rescale_bridge = runner._build_rescale_bridge(train_cfg, log=log)

    ss_baseline_obj = load_static_skeletons_baseline(
        rescale_optimizer_root=str(train_cfg.inproc_rescale_optimizer_root),
        dataset=str(train_cfg.profile),
        num_layers=int(ev.total_layers),
        gelu_per_layer=[int(x) for x in fixed_gelu.reshape(-1)],
        softmax_per_layer=[int(x) for x in fixed_softmax.reshape(-1)],
    )
    ss_action_vec, max_sfs, ss_cost_stats, _ss_diag = static_skeletons_baseline_to_action(
        ss_baseline_obj,
        snap_sf_to_noise_table=False,
    )
    baseline_action_vec = np.asarray(ss_action_vec, dtype=np.int64).reshape(-1)
    log(f"  {bullet} static_skeletons baseline loaded from {ss_baseline_obj.archive_path}")

    # ---------- 3) base env ----------
    base_env = BLBStage2Env(
        handler=ev.reversible_handler,
        model=ev.model,
        probe_batches=probe_batches,
        rescale_bridge=rescale_bridge,
        baseline=BaselineCostStats(),
        reward_weights=RewardWeights(),
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
    base_env.sync_degree_vectors_from_model()

    # ---------- 4) baseline cost / reward weights ----------
    from .env import estimate_baseline_cost_stats
    from .reward import calibrate_weights_from_baseline
    precomputed = {
        "total_bits_sum": int(ss_cost_stats.total_bits_sum),
        "total_fusion_count": int(ss_cost_stats.total_fusion_count),
        "avg_k": float(ss_cost_stats.avg_k),
    }
    baseline = estimate_baseline_cost_stats(
        base_env,
        sample_count=int(train_cfg.calibrate_baseline_samples),
        precomputed_baseline_signals=precomputed,
    )
    base_env.baseline = baseline
    weights = calibrate_weights_from_baseline(baseline)
    base_env.reward_weights = weights

    # baseline accuracy/stability
    baseline_metrics = runner._estimate_baseline_metrics(base_env)
    baseline.loss_mean = float(baseline_metrics.loss_mean)
    baseline.loss_std = float(baseline_metrics.loss_std)
    baseline.metric1_mean = float(baseline_metrics.metric1_mean)
    baseline.metric2_mean = float(baseline_metrics.metric2_mean)
    if not np.isfinite(base_env.stab_threshold):
        base_env.stab_threshold = float(baseline.loss_std) * 1.5 + 1e-3

    _seq_block_title(log, "基线信号（baseline cost / reward / metrics）")
    _seq_log_rounded_box(log, [
        f"成本基线（baseline cost）："
        f"total_bits={baseline.total_bits_sum}, "
        f"fusion={baseline.total_fusion_count}, "
        f"avg_k={baseline.avg_k:.2f}",
        f"指标基线（baseline metrics）："
        f"loss={baseline.loss_mean:.4f}, m1={baseline.metric1_mean:.4f}, m2={baseline.metric2_mean:.4f}",
        f"奖励权重（reward weights）："
        f"w_bits={weights.w_bits:.4g}, w_fusion={weights.w_fusion:.4g}, w_k={weights.w_k:.4g}",
        f"硬约束阈值："
        f"acc_threshold={base_env.acc_threshold:.4f}, stab_threshold={base_env.stab_threshold:.4f}",
        f"static_skeletons archive：{ss_baseline_obj.archive_path}",
    ])

    # ---------- 5) sequential env + policy ----------
    seq_env_cfg = SequentialEnvConfig(
        invalid_penalty=float(getattr(train_cfg, "sequential_invalid_penalty", 1.0)),
        cost_shaping_coeff=float(getattr(train_cfg, "sequential_cost_shaping_coeff", 0.05)),
        fusion_shaping_coeff=float(getattr(train_cfg, "sequential_fusion_shaping_coeff", 0.0)),
        early_terminate_on_invalid=bool(getattr(train_cfg, "sequential_early_terminate_on_invalid", False)),
    )
    seq_env = BLBStage2SequentialEnv(base_env=base_env, env_cfg=seq_env_cfg)

    torch.manual_seed(int(train_cfg.seed))
    np.random.seed(int(train_cfg.seed) % (2**32))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_cfg = SequentialPolicyConfig(
        state_dim=int(seq_env.state_dim),
        max_step_dim=int(seq_env.max_step_dim),
        max_num_levels=6,
        horizon=int(seq_env.horizon),
        num_layers=int(ev.total_layers),
    )
    policy = BLBStage2SequentialPolicy(policy_cfg).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(train_cfg.ppo.lr))
    # Warmstart: bias every action head row toward the largest valid index for
    # that slot kind. Effective indices for SF kinds are levels-1 (largest SF);
    # for K, the all-max index is K_LEVELS.index(max(K_LEVELS)).
    warmstart_applied = False
    if bool(train_cfg.warmstart_baseline_bias):
        try:
            preferred = [policy_cfg.max_num_levels - 1] * policy_cfg.max_step_dim
            policy.apply_preferred_per_step_bias(preferred, gain=float(train_cfg.warmstart_bias_gain))
            warmstart_applied = True
        except Exception as exc:
            log(f"  [warmstart][warning] preferred-per-step bias failed: {exc}")

    _seq_block_title(log, "训练超参与环境设置（Training hyperparameters · sequential per-block）")
    _seq_log_rounded_box(log, [
        f"Sequential env：horizon={seq_env.horizon}    "
        f"max_step_dim={seq_env.max_step_dim}    "
        f"state_dim={seq_env.state_dim}    "
        f"device={str(device)}",
        f"Policy：state_dim={policy_cfg.state_dim}, d_hidden={policy_cfg.d_hidden}, "
        f"head=[{policy_cfg.max_step_dim}×{policy_cfg.max_num_levels}]    "
        f"num_layers={policy_cfg.num_layers}",
        f"PPO：lr={train_cfg.ppo.lr:.6g}    "
        f"clip={train_cfg.ppo.clip_range:.3f}    "
        f"n_epochs={train_cfg.ppo.n_epochs}    "
        f"mb={train_cfg.ppo.minibatch_size}    "
        f"ent_coef={train_cfg.ppo.ent_coef:.4g}    "
        f"value_coef={train_cfg.ppo.value_coef:.4g}",
        f"训练规模：total_episodes={train_cfg.total_episodes}    "
        f"rollout_size={max(1, int(train_cfg.rollout_size))}    "
        f"save_interval={int(train_cfg.save_interval)}    "
        f"calibrate_baseline_samples={int(train_cfg.calibrate_baseline_samples)}",
        f"Reward shaping：invalid_penalty={seq_env_cfg.invalid_penalty:.3g}    "
        f"cost_coeff={seq_env_cfg.cost_shaping_coeff:.3g}    "
        f"fusion_coeff={seq_env_cfg.fusion_shaping_coeff:.3g}    "
        f"early_term_on_invalid={seq_env_cfg.early_terminate_on_invalid}",
        f"Warmstart：bias={warmstart_applied}    "
        f"gain={float(train_cfg.warmstart_bias_gain):.3g}    "
        f"（K 槽偏置略低于 K_max=13，PPO 几个 episode 内会自行修正）",
    ])

    # ---------- 6) optional resume ----------
    # ---------- 6) optional resume ----------
    # Persistent dir's standard live checkpoint name. If the caller didn't pass
    # an explicit resume_checkpoint_path, but a live ckpt exists at the
    # standard location, prefer that automatically (matches the launcher's
    # "same params → auto-resume" contract).
    save_path = os.path.join(blb_progress_dir, "blb_stage2_rl_checkpoint_live.pt")
    effective_resume_path = resume_checkpoint_path
    if (not effective_resume_path) and os.path.isfile(save_path):
        effective_resume_path = save_path
        log(f"  {bullet} 检测到已存在 live checkpoint，自动 resume: {save_path}")

    start_episode = 0
    best_reward = -float("inf")
    best_action_vec: Optional[np.ndarray] = None
    if effective_resume_path and os.path.isfile(effective_resume_path):
        try:
            ckpt = torch.load(effective_resume_path, map_location=device)
            ckpt_variant = str(ckpt.get("rl_variant", "") or "")
            if ckpt_variant and ckpt_variant != "blb_v3_sequential":
                log(
                    f"  [resume][warning] checkpoint at {effective_resume_path} "
                    f"has rl_variant={ckpt_variant!r} (expected 'blb_v3_sequential'); "
                    f"skipping load to avoid policy-shape mismatch. Training will "
                    f"start fresh."
                )
            else:
                if "policy" in ckpt:
                    policy.load_state_dict(ckpt["policy"])
                if "optimizer" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer"])
                start_episode = int(ckpt.get("episode", 0))
                if "best_reward" in ckpt:
                    try:
                        best_reward = float(ckpt["best_reward"])
                    except Exception:
                        best_reward = -float("inf")
                if ckpt.get("best_action") is not None:
                    try:
                        best_action_vec = np.asarray(
                            ckpt["best_action"], dtype=np.int64
                        )
                    except Exception:
                        best_action_vec = None
                log(
                    f"  {bullet} resumed from {effective_resume_path} @ "
                    f"ep={start_episode}    best_reward="
                    f"{('+%.4f' % best_reward) if np.isfinite(best_reward) else 'N/A'}"
                )
        except Exception as exc:
            log(f"  [resume][warning] failed to resume from {effective_resume_path}: {exc}")

    status.set_phase("PPO 训练 (sequential per-block)")

    # ---------- 7) sequential training loop ----------
    ppo = SequentialPPOConfig(
        lr=float(train_cfg.ppo.lr),
        clip_range=float(train_cfg.ppo.clip_range),
        n_epochs=int(train_cfg.ppo.n_epochs),
        minibatch_size=int(train_cfg.ppo.minibatch_size),
        ent_coef=float(train_cfg.ppo.ent_coef),
        value_coef=float(train_cfg.ppo.value_coef),
        max_grad_norm=float(train_cfg.ppo.max_grad_norm),
    )
    # Clamp remaining episodes ≥ 0 — if start_episode already exceeds the
    # total, the loop simply runs 0 episodes and the assembly path emits
    # the final report against the existing best_action_vec from the
    # resumed checkpoint.
    remaining_episodes = max(0, int(train_cfg.total_episodes) - int(start_episode))
    seq_train_cfg = SequentialTrainConfig(
        total_episodes=int(remaining_episodes),
        update_every_n_episodes=max(1, int(train_cfg.rollout_size)),
        log_every_n_episodes=max(1, int(train_cfg.rollout_size)),
        seed=int(train_cfg.seed) + int(start_episode),  # offset seed so resumed runs don't replay the same RNG
        ppo=ppo,
    )

    episode_returns: List[float] = []
    total_episodes_planned = int(train_cfg.total_episodes)
    rollout_avg_window: List[float] = []
    rollout_invalid_window: List[int] = []
    rollout_valid_window: List[int] = []
    rollout_terminal_window: List[float] = []
    ppo_update_counter = [0]   # mutable closure cell

    _seq_log_major_rule(log, "开始 PPO 训练（PPO training start）")
    _seq_log_rounded_box(log, [
        f"总回合：{total_episodes_planned}    "
        f"PPO 窗口：每 {max(1, int(train_cfg.rollout_size))} 回合更新一次    "
        f"checkpoint 间隔：每 {int(train_cfg.save_interval)} 回合",
        f"进度文件：{os.path.join(blb_progress_dir, 'blb_stage2_status.json')}",
        f"checkpoint：{save_path}",
        f"训练曲线：{os.path.join(blb_progress_dir, 'blb_stage2_training_curve.png')}",
        f"详细 trace：第 1 个 episode 会逐 step 打印；之后按窗口汇总",
    ])

    def _format_best_action_snippet(action_vec_arr: Optional[np.ndarray]) -> List[str]:
        """Best-action 简要可读形式：按层 / block 维度汇总动作 index 分布。"""
        if action_vec_arr is None:
            return ["best action 尚未产生（episode_count=0）"]
        try:
            from .action_space import action_dims_for_config
            dims = action_dims_for_config(int(ev.total_layers))
            arr = np.asarray(action_vec_arr, dtype=int).reshape(-1)
            if arr.size != len(dims):
                return [f"best action vec dim mismatch: {arr.size} vs {len(dims)}"]
            # 按层切，最后 1 维是 first_input
            per_layer_w = (len(dims) - 1) // int(ev.total_layers)
            lines: List[str] = []
            for li in range(int(ev.total_layers)):
                seg = arr[li * per_layer_w:(li + 1) * per_layer_w].tolist()
                # 简要：印出每层 action_idx 的总和 + 极值 + 一段 idx 列表
                snippet = ", ".join(str(x) for x in seg[:12])
                if len(seg) > 12:
                    snippet += " …"
                lines.append(f"L{li:02d}: sum={sum(seg):3d}  min={min(seg)}  max={max(seg)}  [{snippet}]")
            lines.append(f"first_input_sf idx = {int(arr[-1])}")
            return lines
        except Exception as exc:
            return [f"<format_best_action_snippet failed: {exc}>"]

    def _step_callback(episode_idx: int, step_within: int, info: Dict[str, Any]) -> None:
        """Per-step trace — only for the first episode of the WHOLE run. Helps
        verify wiring and convergence on the very first rollout, then
        gracefully stays silent thereafter (the periodic boxes carry summary)."""
        if int(episode_idx) != 0:
            return
        valid = bool(info.get("valid", True))
        flag = "✓" if valid else "✗"
        step = int(info.get("step", step_within))
        block_idx = int(info.get("block_idx", 0))
        layer_idx = int(info.get("layer_idx", 0))
        total_bits = int(info.get("total_bits", 0))
        fusion = int(info.get("fusion_count", 0))
        reward = float(info.get("reward", 0.0))
        value = float(info.get("value", 0.0))
        action = info.get("action", [])
        action_str = ",".join(str(int(a)) for a in (action[:8] if isinstance(action, list) else []))
        if isinstance(action, list) and len(action) > 8:
            action_str += "…"
        log(
            f"    [ep0 step {step:02d}] {flag} L{layer_idx:02d}-B{block_idx} "
            f"a=[{action_str}] r={reward:+.4f} V={value:+.3f} "
            f"bits={total_bits} fusion={fusion}"
        )

    def _episode_callback(record: EpisodeRecord) -> None:
        nonlocal best_reward, best_action_vec
        episode_returns.append(float(record.total_reward))
        rollout_avg_window.append(float(record.total_reward))
        rollout_invalid_window.append(int(record.invalid_steps))
        rollout_valid_window.append(int(record.valid_step_count))
        rollout_terminal_window.append(float(record.terminal_reward))

        # First episode summary on its own line
        if record.episode_idx == 0:
            log(
                f"  [ep0 summary] total_return={record.total_reward:+.4f}  "
                f"terminal_reward={record.terminal_reward:+.4f}  "
                f"per_step_sum={record.per_step_reward_sum:+.4f}  "
                f"valid={record.valid_step_count}/{record.steps_taken}  "
                f"invalid={record.invalid_steps}  "
                f"first_invalid="
                f"{'-' if record.first_invalid_step is None else f'step{record.first_invalid_step} (L{record.first_invalid_layer}-B{record.first_invalid_block})'}"
            )

        # Track best
        is_new_best = float(record.total_reward) > best_reward
        if is_new_best:
            best_reward = float(record.total_reward)
            best_action_vec = np.asarray(seq_env._pending_full_vec, dtype=np.int64).copy()
            # Episode-new-best banner (v2-style: "回合 N · 训练过程新高")
            _seq_block_title(
                log,
                f"回合 {start_episode + record.episode_idx + 1} · 训练过程新高（episode new best）",
            )
            log(
                f"  {bullet} 总回报 total_reward={record.total_reward:+.4f}  │  "
                f"终局奖励 terminal_reward={record.terminal_reward:+.4f}  │  "
                f"valid_steps={record.valid_step_count}/{record.steps_taken}  │  "
                f"invalid_steps={record.invalid_steps}"
            )
            log("  " + bullet + " 当前 best action vec（按层概览）：")
            for snippet_line in _format_best_action_snippet(best_action_vec):
                log("      " + snippet_line)

        # Periodic checkpoint
        if (record.episode_idx + 1) % int(train_cfg.save_interval) == 0:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                payload = {
                    "policy": policy.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "episode": int(start_episode + record.episode_idx + 1),
                    "best_reward": float(best_reward),
                    "best_action": (
                        best_action_vec.tolist() if best_action_vec is not None else None
                    ),
                    "rl_variant": "blb_v3_sequential",
                }
                tmp = save_path + ".tmp"
                torch.save(payload, tmp)
                os.replace(tmp, save_path)
                log(
                    f"  [checkpoint] 已保存 · 回合 {start_episode + record.episode_idx + 1} "
                    f"→ {save_path}"
                )
            except Exception as exc:
                log(f"  [save_checkpoint][警告] 保存 {save_path} 失败: {exc}")

        try:
            status.update_after_episode(
                int(start_episode + record.episode_idx + 1),
                float(record.total_reward),
                {
                    "invalid_steps": int(record.invalid_steps),
                    "valid_step_count": int(record.valid_step_count),
                    "early_terminated": bool(record.early_terminated),
                    "steps_taken": int(record.steps_taken),
                    "terminal_reward": float(record.terminal_reward),
                    "best_reward": float(best_reward),
                },
            )
        except Exception:
            pass

    def _ppo_update_end_callback(
            metrics: Dict[str, float],
            completed_episodes: int,
            last_record: EpisodeRecord,
            ) -> None:
        ppo_update_counter[0] += 1
        # Per-PPO-update window summary box
        win_n = max(1, len(rollout_avg_window))
        avg_ret = float(np.mean(rollout_avg_window)) if rollout_avg_window else 0.0
        max_ret = float(np.max(rollout_avg_window)) if rollout_avg_window else 0.0
        min_ret = float(np.min(rollout_avg_window)) if rollout_avg_window else 0.0
        avg_inv = float(np.mean(rollout_invalid_window)) if rollout_invalid_window else 0.0
        avg_valid = float(np.mean(rollout_valid_window)) if rollout_valid_window else 0.0
        avg_term = float(np.mean(rollout_terminal_window)) if rollout_terminal_window else 0.0
        _seq_log_rounded_box(log, [
            f"PPO 窗口摘要 · 截至回合（through episode） "
            f"{start_episode + completed_episodes}",
            f"窗口 N={win_n} 回合 ·  "
            f"平均回报 mean return={avg_ret:+.4f} (min={min_ret:+.4f}, max={max_ret:+.4f})  ·  "
            f"平均终局 mean terminal={avg_term:+.4f}",
            f"平均 valid steps={avg_valid:.2f}/{last_record.steps_taken}  ·  "
            f"平均 invalid={avg_inv:.2f}  ·  "
            f"当前 best={best_reward:+.4f}",
            f"policy_loss={metrics.get('policy_loss', 0.0):+.4f}  ·  "
            f"value_loss={metrics.get('value_loss', 0.0):+.4f}  ·  "
            f"entropy={metrics.get('entropy', 0.0):+.4f}  ·  "
            f"clip_fraction={metrics.get('clip_fraction', 0.0):.3f}",
            f"LR={optimizer.param_groups[0]['lr']:.6f}  ·  "
            f"更新序号 update#{ppo_update_counter[0]}  ·  "
            f"PPO 样本数={int(metrics.get('n_samples', 0))}",
        ])
        # clear the rolling window for the next PPO interval
        rollout_avg_window.clear()
        rollout_invalid_window.clear()
        rollout_valid_window.clear()
        rollout_terminal_window.clear()

        # Periodic big progress box (every SEQ_PROGRESS_BOX_PPO_INTERVAL updates,
        # or on the last update of the run).
        is_last_update = (
            start_episode + completed_episodes >= total_episodes_planned
        )
        if (
            ppo_update_counter[0] % SEQ_PROGRESS_BOX_PPO_INTERVAL == 0
            or is_last_update
        ):
            elapsed_now = time.time() - t_start
            done_this_run = float(completed_episodes)
            avg_ep_time = elapsed_now / max(done_this_run, 1.0)
            remaining_ep = float(total_episodes_planned - (start_episode + completed_episodes))
            eta_seconds = avg_ep_time * max(remaining_ep, 0.0)
            progress_lines = [
                f"BLB Stage-2 噪声 RL 进度 · 回合 "
                f"{start_episode + completed_episodes} / {total_episodes_planned}",
                _seq_progress_bar(
                    int(start_episode + completed_episodes),
                    int(total_episodes_planned),
                ),
                f"训练期最优（train best）得分: {best_reward:+.4f}",
                f"已用时: {_seq_fmt_elapsed(elapsed_now)}    "
                f"预计剩余: {_seq_fmt_elapsed(eta_seconds)}    "
                f"预计完成: {_seq_fmt_eta_finish(eta_seconds)}    "
                f"PPO 更新: {ppo_update_counter[0]} 次",
            ]
            _seq_log_rounded_box(log, progress_lines)

    t_start = time.time()
    seq_result = train_sequential(
        env=seq_env,
        policy=policy,
        train_cfg=seq_train_cfg,
        device=device,
        on_episode_end=_episode_callback,
        on_ppo_update_end=_ppo_update_end_callback,
        on_step_end=_step_callback,
        capture_step_infos=False,  # save memory; we surface aggregates instead
        logger=logging.getLogger("blb_stage2_rl.sequential"),
    )
    elapsed = float(time.time() - t_start)
    status.set_phase("已完成")

    _seq_log_major_rule(log, "PPO 训练结束（Sequential PPO training completed）")
    final_invalid_rate = float(seq_result.get("final_invalid_rate", 0.0))
    _seq_log_rounded_box(log, [
        f"总耗时（elapsed）：{_seq_fmt_elapsed(elapsed)}",
        f"完成回合数：{len(episode_returns)}（计划 {total_episodes_planned}）",
        f"PPO 更新次数：{ppo_update_counter[0]}",
        f"最近 10 回合 invalid 率：{final_invalid_rate*100:.1f}%",
        f"训练期最优 best_reward：{best_reward:+.4f}",
        (
            "训练期最优 best_action vec 已确定（写入 result['blb_v3_best_action_vec']，"
            "下游 final-eval 会安装该动作）"
            if best_action_vec is not None
            else "训练期最优 best_action vec 尚未产生"
        ),
    ])

    # ---------- 8) Training curve PNG/NPZ ----------
    try:
        curve_paths = write_training_curves(
            blb_progress_dir,
            episode_returns=episode_returns,
            best_reward_curve=[float(best_reward)] * len(episode_returns) if episode_returns else [],
            ppo_loss_curve=[float(m.get("policy_loss", 0.0)) for m in seq_result.get("ppo_metrics", [])],
            log_fn=log,
        )
        if curve_paths.get("png"):
            log(f"  {bullet} 训练曲线 PNG → {curve_paths['png']}")
    except Exception as exc:
        log(f"  [警告] 写训练曲线失败：{exc}")

    # ---------- 9) Restore handler to clean polynomial-only state ----------
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

    # ---------- 10) Assemble legacy-compat result dict ----------
    cost_reference_noise_config = ev._get_max_noise_configuration()
    cost_reference_tot_c, _ = ev.get_noise_simulated_cost(**cost_reference_noise_config)
    legacy_best = _build_legacy_compatible_best_noise_config(ev)

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
        "limit_loss": float(limit_loss),
        "limit_p": float(limit_p),
        "limit_s": float(limit_s),
        "proxy_limit_loss": float(limit_loss),
        "proxy_limit_p": float(limit_p),
        "proxy_limit_s": float(limit_s),
        "proxy_base_loss": float(base_loss),
        "proxy_base_p": float(base_p),
        "proxy_base_s": float(base_s),
        "raw_model_baseline_metrics": {
            "loss": float(base_loss),
            "metric1": float(base_p),
            "metric2": float(base_s),
        },
        "search_limits": {"loss": float(limit_loss), "metric1": float(limit_p), "metric2": float(limit_s)},
        "status": "completed",
        "training_eval_split": str(ev.get_reward_reference_split_name()),
        "best_noise_config": {k: v.copy() for k, v in legacy_best.items()},
        "stable_search_best_noise_config": {k: v.copy() for k, v in legacy_best.items()},
        "stable_joint_best_noise_config": {k: v.copy() for k, v in legacy_best.items()},
        "selection_diagnostics": {
            "selection_mode": "blb_v3_sequential_runtime_best",
            "best_reward": float(best_reward),
            "best_action_vec": (
                best_action_vec.tolist() if best_action_vec is not None else None
            ),
        },
        "blb_v3_best_action_vec": (
            best_action_vec.tolist() if best_action_vec is not None else None
        ),
        "blb_v3_best_reward": float(best_reward),
        "blb_v3_profile": str(train_cfg.profile),
        "blb_v3_total_episodes": int(train_cfg.total_episodes),
        "rl_variant": "blb_v3_sequential",
        "sequential_diagnostics": {
            "horizon": int(seq_env.horizon),
            "max_step_dim": int(seq_env.max_step_dim),
            "state_dim": int(seq_env.state_dim),
            "episode_count": int(len(episode_returns)),
            "final_invalid_rate": float(seq_result.get("final_invalid_rate", 0.0)),
            "elapsed_sec": float(elapsed),
            "ppo_metric_count": int(len(seq_result.get("ppo_metrics", []))),
        },
        "training_hparams": {
            "blb_v3_total_episodes": int(train_cfg.total_episodes),
            "blb_v3_rollout_size": int(train_cfg.rollout_size),
            "blb_v3_ppo_lr": float(train_cfg.ppo.lr),
            "blb_v3_sequential_rl": True,
            "blb_v3_sequential_invalid_penalty": float(seq_env_cfg.invalid_penalty),
            "blb_v3_sequential_cost_shaping_coeff": float(seq_env_cfg.cost_shaping_coeff),
            "blb_v3_sequential_fusion_shaping_coeff": float(seq_env_cfg.fusion_shaping_coeff),
            "blb_v3_sequential_early_terminate_on_invalid": bool(
                seq_env_cfg.early_terminate_on_invalid
            ),
            "k_trials": int(train_cfg.num_trials_per_step),
            "probe_size": int(getattr(ev, "stage2_probe_size", 256)),
        },
        "reward_diagnostics": {
            "episode_return_mean": (
                float(np.mean(episode_returns)) if episode_returns else None
            ),
            "episode_return_max": (
                float(np.max(episode_returns)) if episode_returns else None
            ),
            "best_reward": float(best_reward),
        },
    }
    return result
