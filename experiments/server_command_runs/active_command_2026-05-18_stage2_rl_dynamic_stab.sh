#!/usr/bin/env bash
set -e

# ----------------------------------------------------------------------
# 1) 优雅停掉前一轮（如果还在跑）。如果没有 rl.pid 就直接跳。
# ----------------------------------------------------------------------
RUN_DIR="Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.005_s2t0.005_s2st0.005"
PIDFILE="$RUN_DIR/rl.pid"
if [ -f "$PIDFILE" ]; then
  RL_PID="$(cat "$PIDFILE")"
  if [ -n "$RL_PID" ] && kill -0 "$RL_PID" 2>/dev/null; then
    echo "[stop-rl] running RL pid=$RL_PID, sending SIGINT (graceful) ..."
    kill -INT "$RL_PID" 2>/dev/null || true
    for i in 1 2 3 4 5 6; do sleep 10; kill -0 "$RL_PID" 2>/dev/null || break; done
    if kill -0 "$RL_PID" 2>/dev/null; then
      echo "[stop-rl] still alive after 60s, sending SIGTERM ..."
      kill -TERM "$RL_PID" 2>/dev/null || true
      for i in 1 2 3; do sleep 10; kill -0 "$RL_PID" 2>/dev/null || break; done
    fi
    if kill -0 "$RL_PID" 2>/dev/null; then
      echo "[stop-rl] still alive after 90s, hard-killing with SIGKILL ..."
      kill -KILL "$RL_PID" 2>/dev/null || true
      sleep 3
    fi
    echo "[stop-rl] previous RL process stopped."
  else
    echo "[stop-rl] no running RL process at pid=$RL_PID."
  fi
else
  echo "[stop-rl] no rl.pid found; skipping."
fi

# ----------------------------------------------------------------------
# 2) --fresh 重跑 BLB Stage-2 sequential RL。
#
#    上一轮（commit 42cfbe4 后）虽然 invalid_chain 被 mask 拦光了（window_mean_invalid
#    = 0 全程），但 terminal_reward 仍恒在 -160 ± 50 区间。
#    诊断结论：所有 episode 都跌进 priority-2(stability)，根本到不了 P3(cost)：
#      · noisy baseline loss_std = 0.0048，stab_threshold 兜底 = 0.05
#      · 任何"动一动"的 RL 候选 loss_std ≈ 1（3 trials 的 std 误差就这么大）
#      · P2 公式 r = -50 + (0.05 - 1.0)*100 = -145，每个 episode 都长这样
#      · cost 项 r_bits / r_fusion / r_k 完全看不见，PPO 学不到 cost 维度
#      · diagnostic 标签 "P3(cost)" 是 sequential_runner.py:1253 硬编码骗人的
#
#    本次 4 项关键修复（commit 待定 → push 后服务器 pull 即生效）：
#
#      (a) **动态推导 stab_threshold**（主修复）。
#          训练前采 5 个随机 valid action 跑真 forward，取 loss_std 的 P90
#          作为 stab_threshold（floor=0.5，ceiling=5.0）。这样 typical 候选的
#          loss_std (~1.0) 落在 P3(cost) 区，只有真的 outlier (~2+) 才 P2。
#          baseline 单点 preflight 不再用来定阈值，它太保守。
#      (b) **num_trials_per_step 3 → 5**。
#          3 trials 的 loss_std 相对误差 50%，一个 outlier trial 就把 std 拉到 1+。
#          5 trials 相对误差降到 ~35%，std 估计更鲁棒。代价：terminal forward
#          时间 +67%，总训练 wall time +30%。
#      (c) **修复 sequential_runner.py:1253 fake P3(cost) 标签**。
#          原来是 `priority = 1 if invalid_steps>0 else 3` —— 和真正的
#          breakdown.priority 完全没关系。改成从 EpisodeRecord.terminal_priority
#          读真值。details/ 文件里 "priority=P2(stab)" 才意味着真在 P2。
#      (d) **EpisodeRecord 加 terminal_loss_mean / loss_std / metric1_mean 字段**。
#          details/ 文件每条 episode 现在带 "terminal_metrics: loss_mean=X.XX
#          loss_std=X.XX m1=X.XX" 一行，下次跑完可以直接验证 reward 来自哪条
#          优先级、loss_std 是不是真的高、acc 有没有被踩。
#
#    保留不动（上一轮已经 OK 的）：
#      · ForbiddenActionMask + rejection-sample（commit 42cfbe4）
#      · invalid_chain 跳过 forward（commit 42cfbe4）
#      · Warmstart bias 3.5 + preferred_index=LEVELS_F-1=4（commit 42cfbe4）
#      · per-block invalid 可见性（commit f507f25）
#      · noisy baseline preflight（commit 173596d）—— 现在只用来定 acc_threshold
# ----------------------------------------------------------------------
echo ""
echo "================================================================================"
echo "Start BLB Stage-2 Sequential RL (fresh) — dynamic stab calib + 5 trials + real priority label"
echo "================================================================================"
bash llama_7B_LayerImportance.sh run rl --preset mrpc-blb-stage2-rl --fresh
