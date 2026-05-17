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
```

---

## 元信息（meta，给人看的，agent 忽略）

- **任务**：在 mrpc-blb-stage2-rl preset 下 fresh 跑一轮 6000-episode sequential RL，验证最新的三组修复
- **更新时间**：2026-05-17
- **更新原因**：之前一轮 (`f507f25` 之前的) RL 因为 reward 恒 -150 + 没有 invalid 屏蔽，前 215 个 episode 的 best action 在 final-eval 里有 8/59 个 block invalid，模型 Acc 跌到 0.316。本次三组修复全部 push 上来后再跑一次：
    1. (commit `173596d`) noisy baseline preflight → reward 不再塌成常数；
    2. (commit `f507f25`) per-block invalid 可见性 + diagnostic 脚本；
    3. (本次 commit) **ForbiddenActionMask** + rejection-sample + **跳过 invalid 时的 forward** + **warmstart bias 调强**。
- **预期效果**：
    - terminal_reward 应该有显著差分（不再全是 -150）。
    - 每个 PPO 窗口的 `平均 invalid` 应该接近 0（因为 invalid 的 action 在送到 optimizer 之前就被 mask reject 了；如果还能有 invalid，意味着 commit_step 用了 baseline fallback —— 这本身是合法分支但应该罕见）。
    - 训练日志里能看到 `forbidden_action_mask total=N` 计数随训练上升（典型几百到几千）。
    - 训练日志里的 `[checkpoint] 已保存` 行末尾会带 `forbidden_action_mask total=N (top 5: L01-B1=12; ...)`。
- **预期产物**：
    - `Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.005_s2t0.005_s2st0.005/stage2_noise/progress/`
        - `blb_stage2_status.json` —— 实时状态板
        - `diagnostics/diagnostics_summary.md` —— 中文诊断摘要
        - `blb_stage2_rl_checkpoint_live.pt` —— policy + optimizer + forbidden_mask_records
    - `Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.005_s2t0.005_s2st0.005/stage2_noise/`
        - `details/noise_ppo_step_info_<a>-<b>.txt` —— 每 360 回合一个，per-episode 详情
        - `warning.txt` —— 奖励暴跌警告
        - `pruning_search_log.txt` —— 主日志
- **预期耗时**：6000 episodes × 59 sub-steps × (每 sub-step 1 个 optimizer call + 至多几次 rejection)，约 6-7 小时（与之前同量级；rejection 是 ms 级的，不显著拖慢）。

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
