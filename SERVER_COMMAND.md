# 服务器端运行控制文件（SERVER_COMMAND.md）

> **协议**：服务器 agent 监听本文件 → 提取**第一个 ```bash 代码块** → 在仓库根目录 `bash` 执行。
> 本地这边改一次 + push，远端下次同步 / 触发就会按新命令跑。下方 metadata 段只是给人看的，agent 不解析。

## ▶ active command

```bash
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
```

---

## 元信息（meta，给人看的，agent 忽略）

- **任务**：停掉 buggy reward 的 RL 训练 → 跑两次 final-eval（baseline + 当前 best）→ 给 reward-fix 后 RL 的对照基线
- **更新时间**：2026-05-17 凌晨 02:30
- **更新原因**：上一轮 RL 跑了 215 episodes，发现 **terminal_reward 恒 = -150.0000**（不是 -149.9，是 EXACTLY -150）。
  根因：
    1. `sequential_runner.py` 用 `baseline.loss_std * 1.5 + 1e-3` 推 `stab_threshold`，但 `baseline.loss_std`
       来自 **clean** 模型（无 BLB 噪声装配 → K trials 完全 deterministic → std = 0），所以 `stab_threshold = 0.001`。
    2. 任意 RL 候选动作装了 BLB 噪声后，K trials 的 `loss_std` 自然 > 0.001 → 永远命中 priority-2 stability 罚。
    3. 进一步，cross_entropy 在重噪声下可能给出 `inf` loss，`np.std` 传播 inf，落到 `compute_reward`
       priority-2 的 fallback 分支：`r = -50 - 1.0 * 100 = -150`（EXACT）。
    4. 结果：所有候选动作的 terminal_reward 都是 -150，PPO 收不到差分信号，整个搜索空间塌成同一值。
  Commit（待）已在 `blb_stage2_rl/sequential_runner.py` 补回了 legacy single-shot 路径有但 sequential 漏的
  **noisy baseline preflight**（用 baseline action 装真噪声跑一次 K trials → 拿 noisy `loss_std` /
  `metric1_mean` → 校准 `acc_threshold = noisy_acc - limit_tol`、`stab_threshold = noisy_std * (1 + stab_tol) + 1e-3`，
  并设置 `≥ 0.05` 的 floor），并在 `blb_stage2_rl/env.py:_eval_on_probe` 加了 inf/nan 裁剪（cap 100）防止
  单 trial 数值溢出污染整个 episode 的 std。回归在 `tests/test_sequential_smoke.py::OutputHygieneRegressionTest`。
  下一次 `--fresh` RL 跑就是修复后的版本。

### 这一次脚本会做什么

| 步骤 | 命令 | 说明 |
|------|------|------|
| 1 | `kill -INT / -TERM / -KILL` running RL pid | 优雅 → 强制 → 硬杀，三段递进，最多等 90s |
| 2 | `Paean/run_final_eval.sh --preset mrpc-blb-baseline-fixed` | **全 max baseline action** 跑 50 trials，对应"理论最稳的 BLB 配置" |
| 3 | `Paean/run_final_eval.sh --preset mrpc-final-eval-only --action-config best.json` | **当前 RL 训练期 best**（ep203, total_reward=-155.44, 56/59 valid）跑 50 trials |

### 预期产物

| 路径 | 内容 |
|------|------|
| `Paean/outputs/rl/bert-base/mrpc/<slug>/final_eval/baseline_blb_s1t0.005/` | baseline 50-trial 指标聚合（mean / std / min / max accuracy & F1） |
| `Paean/outputs/rl/bert-base/mrpc/<slug>/final_eval/rl_best_ep203_buggy_reward/` | RL best 50-trial 指标聚合 |

每个目录里典型有 `report.md`（人类可读）+ `metrics.json`（机器可读）+ `trials/*.json`（每个 trial 原始数据）。

### 预期耗时

- 停 RL：≤ 90s。
- 一次 final-eval：50 trials × ~30s/trial ≈ 25 分钟。
- 两次总计：**~55 分钟**（顺序跑，不并发）。

## 切换到其他常用任务时（备查，agent 不读这一段）

需要换任务时，**直接覆盖上面的 active command 代码块** + 改这里的元信息。下面只是常用命令样板，不会被执行：

- 重新跑 RL（修复后版本，6000 episodes）：
  `bash llama_7B_LayerImportance.sh run rl --preset mrpc-blb-stage2-rl --fresh`
- 续训（同 preset 不带 `--fresh`，自动检测持久化目录）：
  `bash llama_7B_LayerImportance.sh run rl --preset mrpc-blb-stage2-rl`
- 多 seed 扫（5 seeds，隔离持久化目录）：
  `bash tools/run_multi_seed.sh mrpc-blb-stage2-rl 1,2,3,4,5 trial1 --fresh`
- 旧 single-shot 路径回退：
  `bash llama_7B_LayerImportance.sh run rl --preset mrpc-blb-stage2-rl --fresh --blb-v3-no-sequential-rl`

## 服务器 agent 期望

- agent 只读这个文件的**第一个 ```bash 代码块**，其余 markdown 全部忽略。
- agent 应该在仓库根目录 `bash` 执行（不要 `cd`，所有路径已经按相对仓库根写好）。
- 如果该文件未变更（git hash 未动），agent 不应重复触发同一命令 —— 由 agent 侧做幂等。
- 本次脚本会主动停掉正在跑的 RL（基于 `<slug>/rl.pid`）；agent 不需要额外的 pre-kill 钩子。
