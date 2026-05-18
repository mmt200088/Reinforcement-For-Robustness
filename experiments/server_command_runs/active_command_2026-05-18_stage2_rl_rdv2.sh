#!/usr/bin/env bash
set -e

# ----------------------------------------------------------------------
# 1) 优雅停掉前一轮（如果还在跑）。检查老/新两条持久化路径下的 rl.pid。
#    新一轮（commit "Sequential RL reward redesign v2"）起，CONSTRAINT_SLUG
#    追加 _rdv2 后缀，所以老路径和新路径都要扫一遍。
# ----------------------------------------------------------------------
stop_rl_at_dir() {
  local PIDFILE="$1/rl.pid"
  [ -f "$PIDFILE" ] || { echo "[stop-rl] $1: no rl.pid"; return 0; }
  local RL_PID
  RL_PID="$(cat "$PIDFILE")"
  if [ -z "$RL_PID" ] || ! kill -0 "$RL_PID" 2>/dev/null; then
    echo "[stop-rl] $1: pid=$RL_PID already dead"
    return 0
  fi
  echo "[stop-rl] $1: running pid=$RL_PID, SIGINT ..."
  kill -INT "$RL_PID" 2>/dev/null || true
  for i in 1 2 3 4 5 6; do sleep 10; kill -0 "$RL_PID" 2>/dev/null || break; done
  if kill -0 "$RL_PID" 2>/dev/null; then
    echo "[stop-rl] $1: still alive after 60s, SIGTERM ..."
    kill -TERM "$RL_PID" 2>/dev/null || true
    for i in 1 2 3; do sleep 10; kill -0 "$RL_PID" 2>/dev/null || break; done
  fi
  if kill -0 "$RL_PID" 2>/dev/null; then
    echo "[stop-rl] $1: still alive after 90s, SIGKILL ..."
    kill -KILL "$RL_PID" 2>/dev/null || true
    sleep 3
  fi
  echo "[stop-rl] $1: stopped."
}

# 老路径（ADR-002 hard-priority -150 stuck reward 那版）
stop_rl_at_dir "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.005_s2t0.005_s2st0.005"
# 新路径（ADR-007 v2-style rdv2 reward）
stop_rl_at_dir "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.005_s2t0.005_s2st0.005_rdv2"

# ----------------------------------------------------------------------
# 2) --fresh 重跑 BLB Stage-2 sequential RL。
#
#    上一轮（commit d0ab4bc，"dynamic stab calibration"）失败：动态校准 loop
#    用均匀随机 action 采样，结果 25 次都是 invalid_chain → calibration aborted
#    → stab_threshold 回落到 0.05 → 和上上轮一样 reward 恒 -212 →
#    1000 episode 全程 P2 stuck。证据：
#    `reports/stage2_rl/failed_runs/2026-05-18_dynamic_stab_calibration_fallback/`
#
#    诊断：reward design 本身有问题，不是阈值校准能修的：
#      · hard-priority -50/-100/-200 大惩罚 → 单点 outlier 拖崩 PPO advantage
#      · 任何 BLB candidate (≠baseline) loss_std ≈ 1+，永远在 P2
#      · 一旦所有 episode 都在 P2，candidate 之间 reward 差别全淹没在大惩罚里
#      · v2 (noise_rl_module_v2.py) 在 stage1 上工作得很好，思路可以借鉴
#
#    本次实施 v2 风格 reward redesign（ADR-007 取代 ADR-002 实现）：
#
#      (a) **clipped shaping + tier_bonus** (主修复)。
#          shaping = margin_acc + cost_score + stab_penalty + invalid_term
#          clipped to [-5, +5]
#          tier_bonus = 0 if not metric_ok
#                     else +20 (metric_ok) + (+20 if stab_ok else 0)
#          total reward in [-5, +45]，PPO advantage 永远 bounded。
#          硬优先级靠 tier_bonus +20/+40 大跳变体现，cost 永远进不了 stab/acc tier，
#          但同时 cost differential 在 clip 范围内仍清晰可见，PPO 能学。
#      (b) **stability = soft continuous penalty**，不是 hard gate。
#          stab_penalty = -lambda_stab × max(0, loss_std - stab_threshold)
#          stab_threshold 用 v2 公式 baseline_loss_std × (1 + tol)，不再校准。
#          loss_std=1.0 → penalty=-5（饱和 clip），但不会跌到 -150。
#      (c) **持久化目录加 _rdv2 后缀**。
#          新 dir: `s1t0.005_s2t0.005_s2st0.005_rdv2`
#          老 dir 的 checkpoint 不会和新代码混。以后再改 reward 设计就 bump _rdv3。
#      (d) **写 ADR-007 取代 ADR-002 实现**（intent 保留：cost 不能越级补偿 acc/stab）。
#
#    保留不动（前几轮已经 OK 的）：
#      · ForbiddenActionMask + rejection-sample（commit 42cfbe4）
#      · invalid_chain 跳过 forward（commit 42cfbe4）
#      · Warmstart bias 3.5 + preferred_index=4（commit 42cfbe4）
#      · per-block invalid 可见性 + diagnostic 脚本（commit f507f25）
#      · noisy baseline preflight（commit 173596d）
#      · num_trials_per_step = 5（commit d0ab4bc）
#      · 修复 fake P3(cost) 标签（commit d0ab4bc）
#      · EpisodeRecord 带 terminal_loss/metric 字段（commit d0ab4bc）
# ----------------------------------------------------------------------
echo ""
echo "================================================================================"
echo "Start BLB Stage-2 Sequential RL (fresh, _rdv2) — v2-style clipped+tier reward (ADR-007)"
echo "================================================================================"
bash llama_7B_LayerImportance.sh run rl --preset mrpc-blb-stage2-rl --fresh
