# 服务器端运行控制文件（SERVER_COMMAND.md）

> **协议**：服务器 agent 监听本文件 → 提取**第一个 ```bash 代码块** → 在仓库根目录 `bash` 执行。
> 本地这边改一次 + push，远端下次同步 / 触发就会按新命令跑。下方 metadata 段只是给人看的，agent 不解析。

## ▶ active command

```bash
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
```

---

## 元信息（meta，给人看的，agent 忽略）

- **任务**：在 mrpc-blb-stage2-rl preset 下 fresh 跑一轮 6000-episode sequential RL，验证 reward design 修复
- **更新时间**：2026-05-18
- **更新原因**：上一轮（commit 42cfbe4）虽然把 invalid_chain 拦光了，但 reward 恒 -160。诊断发现真正的 priority 是 P2(stability)，stab_threshold=0.05 太低，任何 RL 候选 loss_std 都过这条线。本次：
    1. 动态推导 stab_threshold —— 用 5 个随机 valid action 的 loss_std P90 作为阈值，让 typical 候选能进 P3(cost)；
    2. num_trials_per_step 3→5，让 std 估计鲁棒一档；
    3. 修复 details/ 文件的 `priority=P3(cost)` 假标签 —— 它原来是硬编码的，现在读 EpisodeRecord.terminal_priority 真值；
    4. EpisodeRecord 加 loss_std/loss_mean/m1 字段，下一轮 details/ 直接能验证 reward 来自哪条优先级。
- **预期效果**（这次要看的关键信号）：
    - **terminal_reward 应该跨过 0 进入正区间**（baseline 算 +27，best 候选应该 +25 左右；而不是上一轮的 -160）。
    - details/ 文件每条 episode 的 `priority=` 应该绝大多数是 `P3(cost)`，而不是上一轮全是假的 P3。如果还有大量 P2(stab)，说明 stab_threshold 还需再调宽。
    - `terminal_metrics: loss_mean=X loss_std=Y m1=Z` 一行直接能验证：loss_std 应该 < stab_threshold（动态校准后通常 1-2 之间），m1 应该 ≥ acc_threshold。
    - 训练曲线 best_reward 应该真的能优化 cost —— total_bits 该单调下降（从 baseline 14779 → 真优化目标 < 13000）。
    - 上一轮的 ForbiddenActionMask + invalid skip-forward + warmstart bias 继续生效，window_mean_invalid 仍接近 0。
- **预期产物**：
    - `Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.005_s2t0.005_s2st0.005/stage2_noise/progress/`
        - `blb_stage2_status.json` —— 实时状态板
        - `diagnostics/diagnostics_summary.md` —— 中文诊断摘要
        - `blb_stage2_rl_checkpoint_live.pt` —— policy + optimizer + forbidden_mask_records
    - `Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.005_s2t0.005_s2st0.005/stage2_noise/`
        - `details/noise_ppo_step_info_<a>-<b>.txt` —— 每 360 回合一个，per-episode 详情（**这次会带真 priority 标签 + terminal_metrics 行**）
        - `warning.txt` —— 奖励暴跌警告
        - `pruning_search_log.txt` —— 主日志（**这次启动头部能看到 "稳定阈值校准来源: P90 ... = X.XX"**）
- **预期耗时**：trials 3→5 单 episode 时间 +30%；6000 episodes 约 8-9 小时（上一轮 ~6 小时）。setup phase 多 5 个随机 sample × 5 trials ≈ 30s，可忽略。

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
