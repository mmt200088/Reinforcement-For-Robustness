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
```

---

## 元信息（meta，给人看的，agent 忽略）

- **任务**：在 mrpc-blb-stage2-rl preset 下 fresh 跑一轮 6000-episode sequential RL，验证 warmstart hotfix
- **更新时间**：2026-05-18（晚）
- **更新原因**：上一轮（commit `4097bea`, warmstart hotfix）anchor 工作正常（reward +36.77），但 sample 阶段一开始 reward 立刻塌到 -7.8。bug 报告：`reports/stage2_rl/bug_reports/2026-05-18_warmstart_hotfix_sampling_collapse/`。
    根因：PPO 的 entropy bonus（ent_coef=0.02）在 anchor 阶段把 policy 越拉越散。3 次 anchor PPO update 的 entropy 从 6.48 涨到 9.21（接近 13-slot 均匀的最大值）。entropy 项的梯度 (-0.02 × 9.21 = -0.18) 比 policy_grad (+0.08) 还大 2x，导致 PPO 整体把 policy 推向"最大化 entropy"而不是收敛 baseline。Sample 一开始就发散 → 多 slot 偏离 baseline → acc 跌穿。
- **本次改动汇总**：
    1. `blb_stage2_rl/sequential_policy.py` `sequential_ppo_update` 新增 `ent_coef_override` 参数，PPO update 期间用调度后的 ent_coef 替换 cfg.ent_coef；metrics dict 新增 `ent_coef` 字段。
    2. `blb_stage2_rl/sequential_runner.py` 新增 `_resolve_ent_coef_schedule(...)` 帮手：anchor 期返回 0.0，ramp 期线性插值，steady 期返回 target；`train_sequential` 每次 PPO update 前算 current_ent_coef 并传过去。
    3. `SequentialTrainConfig` + `BLBStage2TrainConfig` 加 `ent_coef_anchor=0.0` 和 `ent_coef_ramp_episodes=240` 默认值。
    4. PPO 更新摘要 + diagnostics_summary.md 的 PPO 表格都加 `ent_coef` 列；启动 box 加 entropy schedule 说明行。
    5. 测试：`EntCoefScheduleRegressionTest`（7 个新 case，含 helper 的 functional anchor / ramp / steady test）。40/40 smoke tests pass。
- **预期效果**（这次要看的信号）：
    - **anchor 期（eps 0-119）entropy 应该下降不再上升**（无 entropy bonus，policy_grad 单独把 policy 集中到 baseline）；window_mean_return ~+36-40。
    - **PPO update 摘要 + diagnostics PPO 表格里 `ent_coef` 列**：updates 1-2 都是 0.00000；update 3 起开始 ramp（每个 update +0.005 左右）；update 6 之后 steady 在 0.02。
    - **sample 期（eps 120+）reward 应该 ≥ +20**（policy 已集中 baseline，sampled actions 多数接近 baseline → acc 仍在阈值上方）。
    - 训练后期 PPO 在 baseline 邻域降 cost：reward 可能突破 +45。
    - 每次找到新 best 时，日志里有 `推理指标 ... loss_mean=X loss_std=Y m1=Z priority=P3(cost)` 一行（commit 4097bea 加的，本次保留）。
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
