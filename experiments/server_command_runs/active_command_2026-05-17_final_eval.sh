set -e

# ----------------------------------------------------------------------
# 1) 先优雅停掉正在跑的（buggy reward 的）RL 训练
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
    echo "[stop-rl] RL process stopped."
  else
    echo "[stop-rl] no running RL process at pid=$RL_PID."
  fi
else
  echo "[stop-rl] no rl.pid found; skipping."
fi

# ----------------------------------------------------------------------
# 2) Final-eval #1 —— BLB 全 max baseline 配置（safe reference）
#    输出: Paean/outputs/rl/bert-base/mrpc/<slug>/final_eval/baseline_blb_s1t0.005/
# ----------------------------------------------------------------------
echo ""
echo "================================================================================"
echo "FINAL EVAL #1 / 2 · BLB baseline action (all-max SFs, per-block max K)"
echo "================================================================================"
bash Paean/run_final_eval.sh \
    --preset mrpc-blb-baseline-fixed \
    --run-name baseline_blb_s1t0.005

# ----------------------------------------------------------------------
# 3) Final-eval #2 —— 当前 RL 训练期 best 配置（episode 203，total_reward=-155.4）
#    注意：该 best 是在**有 bug 的 reward**（terminal 恒 -150）下选出来的，
#    所以并不一定对应真正最优的 BLB 配置。本次 eval 主要是给一个 "buggy RL
#    在 200 episode 内得到的局部最优" 的 final 数字，作为后续 reward-fix
#    后 RL run 的对照基线。
#    输出: Paean/outputs/rl/bert-base/mrpc/<slug>/final_eval/rl_best_ep203_buggy_reward/
# ----------------------------------------------------------------------
BEST_JSON="Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.005_s2t0.005_s2st0.005/stage2_noise/progress/diagnostics/best_action_vec.json"
if [ ! -f "$BEST_JSON" ]; then
  echo "[FATAL] best_action_vec.json not found at: $BEST_JSON"
  exit 1
fi
echo ""
echo "================================================================================"
echo "FINAL EVAL #2 / 2 · RL training best @ ep203 (buggy reward)"
echo "================================================================================"
bash Paean/run_final_eval.sh \
    --preset mrpc-final-eval-only \
    --action-config "$BEST_JSON" \
    --run-name rl_best_ep203_buggy_reward

echo ""
echo "================================================================================"
echo "DONE · 结果在 Paean/outputs/rl/bert-base/mrpc/<slug>/final_eval/{baseline_blb_s1t0.005,rl_best_ep203_buggy_reward}/"
echo "================================================================================"
