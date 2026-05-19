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
#    上一轮（commit 4097bea, warmstart hotfix）anchor 阶段（eps 0-119）
#    reward = +36.77 ✓ baseline 执行成功；但切到 PPO sample 后（eps 120+）
#    reward 立刻塌到 -7.78（terminal=-5 clip 底）。bug 报告：
#    `reports/stage2_rl/bug_reports/2026-05-18_warmstart_hotfix_sampling_collapse/`
#
#    根因（用 PPO update entropy 数据印证）：
#      · 3 次 anchor PPO update 的 entropy：6.48 → 8.47 → 9.21（持续上升）
#      · PPO loss = policy_grad - ent_coef × entropy，ent_coef=0.02
#      · update 3: policy_grad=+0.083，entropy 项 = -0.02 × 9.21 = -0.184
#      · entropy 项比 policy_grad 大 2 倍以上 → 梯度被"最大化 entropy"主导
#      · 整个 anchor 期间 PPO 不是让 policy 收敛 baseline，反而越来越发散
#      · sample 阶段一开始，发散的 policy 立即偏离 baseline 多 slot →
#        acc 跌破阈值 → metric_ok=False → reward=-5 clip 底
#
#    本次单项修复（entropy schedule，commit 待定）：
#
#      (a) **anchor 期 ent_coef=0**：forced-baseline 阶段完全关掉 entropy bonus，
#          让 policy_grad 单独把 policy 集中到 baseline 上（policy_loss 是
#          负的→push baseline 概率上升）。
#      (b) **sample 期前 240 episode 线性 ramp ent_coef 从 0 → 0.02**：
#          ramp 完成后回到原本的 0.02 steady。给 PPO 一个 "先稳，再探索" 的过渡。
#      (c) **日志 + diagnostics 加 ent_coef 列**：PPO update 摘要 + diagnostics_summary.md
#          的 PPO 表格都加 ent_coef 列，跑起来一眼能看 schedule 是否生效。
#
#    保留不动（之前已经 OK 的）：
#      · per-slot mode warmstart bias（commit 4097bea）✓
#      · forced baseline anchor（commit 4097bea）—— anchor 阶段已经能拿 +36.77
#      · v2-style clipped+tier reward（commit b97ca83）
#      · stab_threshold = baseline_loss_std × (1 + tol)（commit b97ca83）
#      · ForbiddenActionMask + rejection-sample（commit 42cfbe4）
#      · 持久化目录单 dir（commit 4097bea）
#      · 新最优日志带 metric 行（commit 4097bea）
# ----------------------------------------------------------------------
echo ""
echo "================================================================================"
echo "Start BLB Stage-2 Sequential RL (fresh) — entropy schedule hotfix (anchor ent_coef=0 + ramp)"
echo "================================================================================"
bash llama_7B_LayerImportance.sh run rl --preset mrpc-blb-stage2-rl --fresh
