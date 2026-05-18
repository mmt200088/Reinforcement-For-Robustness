# 服务器端运行控制文件（SERVER_COMMAND.md）

> **协议**：服务器 agent 监听本文件 → 提取**第一个 ```bash 代码块** → 在仓库根目录 `bash` 执行。
> 本地这边改一次 + push，远端下次同步 / 触发就会按新命令跑。下方 metadata 段只是给人看的，agent 不解析。

## ▶ active command

```bash
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
```

---

## 元信息（meta，给人看的，agent 忽略）

- **任务**：在 mrpc-blb-stage2-rl preset 下 fresh 跑一轮 6000-episode sequential RL，验证 warmstart hotfix
- **更新时间**：2026-05-18（晚）
- **更新原因**：上一轮（commit `b97ca83`，ADR-007 v2 reward）在 _rdv2 目录跑了 200 episode，reward 仍恒在 -7.8。bug 报告 / 诊断报告：
    - `reports/stage2_rl/bug_reports/2026-05-18_stage2_rl_rdv2_negative_reward_startup/`
    - `reports/stage2_rl/fix_reports/2026-05-18_warmstart_acc_collapse_fix/report.html`
    根因：**不是 reward design 问题**（ADR-007 v2 公式没问题）。是 sequential 路径的 warmstart bias preferred=[4]*13 把 13 个 slot 位置里的 8 个偏到错误 index → 8/13 槽位实际是均匀采样 → 343/577 slot 与 baseline 不同 → acc 跌穿 acc_threshold → metric_ok=False → reward = clip 底 -5 → 总 reward ≈ -7.8。
- **本次改动汇总**：
    1. `sequential_runner.py` 新增 `_compute_per_slot_mode_preferred`：对每个 slot 位置在 59 个 step 上取 baseline_action_vec 的众数作为 preferred，替换旧的硬编码 [4]*13。
    2. `sequential_runner.py train_sequential` 新增 `force_baseline_episodes` 参数 + 短路逻辑：前 N=max(60, rollout_size*2) 个 episode 直接执行 baseline action，policy 通过 evaluate_action 写 buffer，value head 学到 +45 baseline reward，policy 概率质量预热到 baseline 附近。
    3. `llama_7B_LayerImportance.sh` 回滚 `_rdv2` 后缀，持久化目录恢复为 `s1t0.005_s2t0.005_s2st0.005`（按用户要求，单目录维护更省心）。
    4. "新最优" 日志加一行推理指标 `loss_mean / loss_std / m1 / priority / total_bits / fusion`，方便复查。
    5. 测试更新：加 `WarmstartFixedRegressionTest`（4 个新 case，含 helper 的功能性 functional test）。33/33 smoke tests pass。
- **预期效果**（这次要看的信号）：
    - **前 60 episode 都是 forced-baseline，reward 全部 ≈ +45**（确定值），日志里能看到 `forced_baseline=True` 标记。
    - **episode 60+ 切到 PPO sample 后，前几十个 sampled 候选 reward 应该落在 [+15, +45]**（policy 已偏向 baseline，与 baseline 相差几个 slot，acc 仍应在阈值上方）。
    - 训练后期 PPO 在 baseline 邻域降 cost：reward 可能突破 +45（cost-better-than-baseline 区，metric+stab 都 ok）。
    - 每次找到新 best 时，日志里会有 `推理指标 ... loss_mean=X loss_std=Y m1=Z priority=P3(cost)` 一行，直接对照 acc_threshold = 0.8653。
    - total_bits 训练后期应稳定下降（baseline 14779 → 目标 12500-13500）。
- **预期产物**（**回滚单目录**）：
    - `Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.005_s2t0.005_s2st0.005/stage2_noise/progress/`
        - `blb_stage2_status.json` —— 实时状态板
        - `diagnostics/diagnostics_summary.md` —— 中文诊断摘要
        - `blb_stage2_rl_checkpoint_live.pt` —— policy + optimizer + forbidden_mask_records
    - `Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.005_s2t0.005_s2st0.005/stage2_noise/`
        - `details/noise_ppo_step_info_<a>-<b>.txt`（带 `terminal_metrics: loss_mean=X loss_std=Y m1=Z`）
        - `warning.txt` —— 奖励暴跌警告
        - `pruning_search_log.txt` —— 主日志（启动头部能看到 `preferred per slot (mode over 59 steps) = [...]` 和 `强制 baseline 锚点: 前 N 个 episode...`）
- **预期耗时**：~8-9 小时。前 60 个 episode 跑 baseline action 也要 5 trials × 4 probe forward 加 59 个 optimizer call，但 PPO update 仍按 60-episode rollout 算。

### Stage-1 → Stage-2 degree 适配（用户问题 #1）

经过审查代码链：
- `_resolve_stage2_fixed_stage1_config` → `resolve_stage1_only` 从 `glue_final_configs_best_ppo.json` 的 `bert-base.mrpc.stage1.gelu/softmax` 读出 per-layer 向量。
- 传给 `BLBStage2RLRunner.run(fixed_gelu, fixed_softmax)` → 传给 `BLBStage2Env(gelu_degree=fixed_gelu, attn_degree=fixed_softmax)`。
- `BLBStage2Env._normalize_degree_vector` 把 length-L 向量直接保留（不会塌成 scalar=4）。
- `evaluate_action_for_cost(..., gelu_degree=self.gelu_degree, attn_degree=self.attn_degree)` 把向量传到 `action_vector_to_cfgs(..., gelu_degree=..., attn_degree=...)`。
- `action_vector_to_cfgs` 每层用 `_degree_for_layer(gelu_degree, li, ...)` 拿出该层的 degree，构造 Block3/5 cfg。
- `make_config_name` 用 `cfg.degree` (Block3) / `cfg.gelu_degree` (Block5) 拼 graph_key，所以每个 (layer, block) 送进 optimizer 时用的是 per-layer graph。
- **Paean/blb_action_eval.py** 走的是同一条 `action_vector_to_cfgs` 路径，per-layer 向量在 line 273-274 显式传入。

→ **训练 / final-eval 都已经在用 per-layer stage-1 degree**。之前 `report.md` 里出现的 `block5_n4` / `block3_exp_n4` 是 **诊断脚本的 bug**（读 JSON 路径错了，回落到 `[4]*12`），不是训练代码的 bug。本次也修了诊断脚本（`scripts/blb_diagnose_invalid_blocks.py:_stage1_degrees_from_meta`）。

如果对训练日志里实际用的 graph_key 还不放心，跑起来之后看 `details/noise_ppo_step_info_*.txt`，每条 `invalid_blocks` 行里都有 `graph=block5_n1_L0` / `graph=block3_exp_n2_L0` 之类字段，能直接验证 per-layer 是否正确。

## 切换到其他常用任务时（备查，agent 不读这一段）

需要换任务时，**直接覆盖上面的 active command 代码块** + 改这里的元信息。下面只是常用命令样板，不会被执行：

- 续训（同 preset 不带 `--fresh`，自动检测持久化目录、恢复 forbidden_mask）：
  `bash llama_7B_LayerImportance.sh run rl --preset mrpc-blb-stage2-rl`
- 单独 final-eval（最新 best）：
  `bash Paean/run_final_eval.sh --preset mrpc-final-eval-only --action-config "$RUN_DIR/stage2_noise/progress/diagnostics/best_action_vec.json"`
- 单独 final-eval（baseline）：
  `bash Paean/run_final_eval.sh --preset mrpc-blb-baseline-fixed --run-name baseline_blb_s1t0.005`
- 离线诊断某个 action 的 invalid_blocks：
  `python scripts/blb_diagnose_invalid_blocks.py --action-config <path> --output-dir reports/blb_opt/invalid_blocks/<name>`
- 多 seed 扫（5 seeds，隔离持久化目录）：
  `bash tools/run_multi_seed.sh mrpc-blb-stage2-rl 1,2,3,4,5 trial1 --fresh`

## 服务器 agent 期望

- agent 只读这个文件的**第一个 ```bash 代码块**，其余 markdown 全部忽略。
- agent 应该在仓库根目录 `bash` 执行（不要 `cd`，所有路径已经按相对仓库根写好）。
- 如果该文件未变更（git hash 未动），agent 不应重复触发同一命令 —— 由 agent 侧做幂等。
- 本次脚本会主动停掉正在跑的 RL（基于 `<slug>/rl.pid`），所以 agent 不需要额外的 pre-kill 钩子。
