# 服务器端运行控制文件（SERVER_COMMAND.md）

> **协议**：服务器 agent 监听本文件 → 提取**第一个 ```bash 代码块** → 在仓库根目录 `bash` 执行。
> 本地这边改一次 + push，远端下次同步 / 触发就会按新命令跑。下方 metadata 段只是给人看的，agent 不解析。

## ▶ active command

```bash
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
```

---

## 元信息（meta，给人看的，agent 忽略）

- **任务**：在 mrpc-blb-stage2-rl preset 下 fresh 跑一轮 6000-episode sequential RL，验证 ADR-007 v2-style reward redesign
- **更新时间**：2026-05-18
- **更新原因**：上一轮（commit `d0ab4bc`，动态校准）再次失败：577-dim 均匀随机 action 25 次全 invalid → 校准 abort → stab_threshold 兜底 0.05 → 重蹈覆辙 (-212 reward)。证据：`reports/stage2_rl/failed_runs/2026-05-18_dynamic_stab_calibration_fallback/`。
    诊断结论：hard-priority -50/-100/-200 大惩罚本质就和 BLB 的 std 噪声水平不匹配，靠校准修不好。
    **方案**：参考 noise_rl_module_v2.py（用户验证过工作良好），重写 reward 为 v2 风格 clipped+tier_bonus（ADR-007）。
- **本次改动汇总**：
    1. `reward.py` 全量重写：`shaping = margin_acc + cost_score + stab_penalty + invalid_term`，clipped 到 [-5, +5]，再叠加 tier_bonus +20(metric_ok) +20(stab_ok)，总范围 [-5, +45]。
    2. `sequential_runner.py` 去掉失败的动态校准 loop，改用 v2 公式 `stab_threshold = baseline_loss_std × (1 + tol)`。stab 现在是 soft penalty，threshold 设紧也不会爆炸。
    3. `llama_7B_LayerImportance.sh` CONSTRAINT_SLUG 加 `_rdv2` 后缀 → 新持久化目录 `s1t0.005_s2t0.005_s2st0.005_rdv2`，老 checkpoint 不会和新代码混。
    4. ADR-002 标记 superseded by ADR-007；ADR-007 完整记录设计 rationale + 失败证据。
    5. 测试更新：BLBRewardRegressionTests 改成断言新四档 reward 范围 ([-5/0/+15/+25/+35/+45])；test_sequential_smoke 加 RewardDesignV2RegressionTest 锁定 v2 字段不被无声 revert。29/29 smoke tests pass。
- **预期效果**（这次要看的信号，每个 episode 都该看到）：
    - **baseline action reward ≈ +45**（baseline 全 max-SF，metric_ok+stab_ok → tier_bonus +40 + clipped shaping ~+5）
    - **典型 RL 候选 reward 在 [+15, +25]**（metric_ok + stab_fail → tier +20，shaping clipped）
    - **罕见 cost-better-than-baseline 候选 reward > +30**（metric_ok + 偶尔 stab_ok）
    - 真的崩的候选 reward < 0（acc_violation 或 invalid 时无 tier_bonus）
    - PPO 看到 ~50 reward 跨度，advantage 信号清晰
    - details/ 每条 episode 带 `priority=P3(cost) terminal_metrics: loss_mean=X loss_std=Y m1=Z`，能直接看
    - total_bits 训练后期应单调下降（从 baseline 14779 → 12500-13500）
- **预期产物**（**注意新路径！老 `s1t0.005_s2t0.005_s2st0.005/` 已废弃**）：
    - `Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.005_s2t0.005_s2st0.005_rdv2/stage2_noise/progress/`
        - `blb_stage2_status.json` —— 实时状态板
        - `diagnostics/diagnostics_summary.md` —— 中文诊断摘要（meta 里 "reward_weights" 应该是 v2-style 字段）
        - `blb_stage2_rl_checkpoint_live.pt` —— policy + optimizer + forbidden_mask_records
    - `Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.005_s2t0.005_s2st0.005_rdv2/stage2_noise/`
        - `details/noise_ppo_step_info_<a>-<b>.txt` —— 每 360 回合一个，per-episode 详情
        - `warning.txt` —— 奖励暴跌警告
        - `pruning_search_log.txt` —— 主日志（启动头部能看到 "v2 formula: noisy_baseline_loss_std=X × (1+tol)=Y"）
- **预期耗时**：~8-9 小时（5 trials × 4 probe batches × 59 sub-steps × 6000 episodes，单 step ~150ms forward）。

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
