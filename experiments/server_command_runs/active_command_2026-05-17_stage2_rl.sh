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
#    本次包含四项关键改动（commit 待定 → push 后服务器 pull 即生效）：
#
#      (a) 在跑训练前 calibrate noisy baseline（acc_threshold / stab_threshold
#          按真噪声 baseline 推得），避免之前 terminal_reward 恒 -150 的 bug；
#          来自 commit 173596d。
#      (b) 优化器 invalid_chain 时跳过模型 forward（env.py），不浪费 GPU。
#      (c) 引入 ForbiddenActionMask + rejection-sample：每个 (layer, block) 在训练
#          过程中累计的"导致 invalid_chain 的 action 元组"被永久拉黑；下次 PPO
#          再采到同样的元组立刻 reject + 重采，直到拿到非黑名单元组才送进 optimizer。
#          ≥32 次重采还都失败则 fallback 到该 step 的 baseline 动作（保证 valid）。
#      (d) Warmstart bias 1.2 → 3.5，且 preferred index 改为 LEVELS_F-1=4（之前 5
#          被 SF 槽位 mask 掉等于没生效）。初始 policy 每个 SF 槽位约 84% 概率落到
#          baseline index，rejection 命中率开局应该很低，PPO 从 baseline 邻域开始探索。
#
#    + per-block invalid 可见性（commit f507f25）：details/ 文件里每个 episode
#      末尾按 "invalid_blocks (N):" 列出每个失败块（在 mask 学好之前主要看这个，
#      mask 接管后 commit-invalid 应基本为 0，但 rejection_counters 里仍能看到
#      "samples_rejected_by_optimizer" / "samples_rejected_by_mask" 计数）。
# ----------------------------------------------------------------------
echo ""
echo "================================================================================"
echo "Start BLB Stage-2 Sequential RL (fresh) — with reward fix + invalid mask + stronger warmstart"
echo "================================================================================"
bash llama_7B_LayerImportance.sh run rl --preset mrpc-blb-stage2-rl --fresh
