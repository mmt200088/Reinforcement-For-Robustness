set -e

# ----------------------------------------------------------------------
# 1) 优雅停掉前一轮（如果还在跑）。同时扫主目录 + 历史 _rdv2 临时目录，
#    防止服务器上有任何残留进程。本次起回滚 _rdv2 后缀，回到单目录形式
#    （用户反馈：多目录维护成本更高，--fresh 强制重启已足够防混用）。
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

# 主目录（本次起 canonical）
stop_rl_at_dir "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.005_s2t0.005_s2st0.005"
# 历史 _rdv2 临时目录（已废弃；如果服务器上还有跑就停掉）
stop_rl_at_dir "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.005_s2t0.005_s2st0.005_rdv2"

# ----------------------------------------------------------------------
# 2) --fresh 重跑 BLB Stage-2 sequential RL。
#
#    上一轮（commit b97ca83，ADR-007 v2 reward）在 _rdv2 目录跑了 200 episode
#    后 reward 仍恒在 -7.8。bug 报告：
#    `reports/stage2_rl/bug_reports/2026-05-18_stage2_rl_rdv2_negative_reward_startup/`
#    诊断报告：
#    `reports/stage2_rl/fix_reports/2026-05-18_warmstart_acc_collapse_fix/report.html`
#
#    根因（**不是** reward design 问题；上一版 ADR-007 reward 没问题）：
#      · sequential 路径的 warmstart bias preferred=[4]*13 把 13 个 slot
#        位置里的 8 个偏到了错误 index（因为不同 slot 位置的 baseline 众数不一样）。
#      · 8/13 槽位 policy 实际是均匀采样 → 343/577 slot 与 baseline 不同 →
#        installed 噪声远比 baseline 重 → acc 跌穿 acc_threshold(0.8653) →
#        每个 episode 都 metric_ok=False → tier_bonus=0 → reward = shaping
#        clip 下限 -5（确认：所有 episode terminal_reward 恰好 -5）。
#
#    本次三项修复（commit "Stage2 RL warmstart hotfix"）：
#
#      (a) **修正 warmstart bias 的 preferred index**（最关键）。
#          替换硬编码 [4]*13 为"对每个 slot 位置在 59 个 step 上取
#          baseline_action_vec 的众数"。13 个 slot 位置现在每个都偏向
#          各自最常见的 baseline 值（4/2/3 三种）。
#      (b) **新增 "forced baseline anchor episodes"** 机制（更强的 warmstart）。
#          前 N=max(60, rollout_size*2) 个 episode 直接执行 baseline action，
#          PPO 通过 policy.evaluate_action 算 log_prob/value 进 buffer，
#          value head 学到 +45 reward，policy 概率质量被推向 baseline。
#          之后再切到 PPO sample 探索，开局就在 baseline 邻域。
#      (c) **持久化目录回滚 _rdv2 后缀**（按用户要求）。
#          回到 `s1t0.005_s2t0.005_s2st0.005`。单目录维护成本低；
#          --fresh 强制重启已经够防混用。
#      (d) **"新最优" 日志补充推理指标行**。
#          找到新 best 时额外打印 loss_mean/loss_std/m1/priority/total_bits/fusion。
#
#    保留不动（之前已经 OK 的）：
#      · v2-style reward formula（commit b97ca83）
#      · stab_threshold = baseline_loss_std × (1 + tol)（commit b97ca83）
#      · ForbiddenActionMask + rejection-sample（commit 42cfbe4）
#      · invalid_chain 跳过 forward（commit 42cfbe4）
#      · per-block invalid 可见性 + diagnostic 脚本（commit f507f25）
#      · num_trials_per_step = 5（commit d0ab4bc）
# ----------------------------------------------------------------------
echo ""
echo "================================================================================"
echo "Start BLB Stage-2 Sequential RL (fresh) — warmstart hotfix: per-slot mode + force-baseline anchor"
echo "================================================================================"
bash llama_7B_LayerImportance.sh run rl --preset mrpc-blb-stage2-rl --fresh
