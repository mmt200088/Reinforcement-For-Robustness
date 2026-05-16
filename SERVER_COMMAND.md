# 服务器端运行控制文件（SERVER_COMMAND.md）

> **协议**：服务器 agent 监听本文件 → 提取**第一个 ```bash 代码块** → 在仓库根目录 `bash` 执行。
> 本地这边改一次 + push，远端下次同步 / 触发就会按新命令跑。下方 metadata 段只是给人看的，agent 不解析。

## ▶ active command

```bash
bash llama_7B_LayerImportance.sh run rl --preset mrpc-blb-stage2-rl --fresh
```

---

## 元信息（meta，给人看的，agent 忽略）

- **任务**：BLB Stage-2 per-block sequential RL 训练，MRPC，bert-base
- **更新时间**：2026-05-17
- **更新原因**：上一次跑的输出目录被改到了 `Paean/outputs/`，且 `pruning_search_log.txt` 因为
  Python 隐式拼接 + 运算符优先级 bug，开头被复制 80 次 banner。commit `2ee7be8`
  已把：
    1. 输出目录调回 `Parting Chapter/persistent/`；
    2. 修了 80x header bug（`layer_importance_evaluator.py:3470`）；
    3. 动作展示一槽一行 + 列对齐；
    4. PPO 框线 `╭─╮│╰╯` 全删，改成短分隔 + bullet；
    5. sequential 路径补回 `details/` + `warning.txt`（与 legacy v2 对齐）。
  所以这次必须 `--fresh` 重跑一次，验证以上五项是否都生效。
- **预期输出根目录**：
  `Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.005_s2t0.005_s2st0.005/blb_stage2/progress/`
- **预期产物**（训练中可 tail）：
    - `blb_stage2_status.json` —— 实时状态板，atomic 写
    - `diagnostics/diagnostics_summary.md` —— 中文诊断摘要
    - `details/noise_ppo_step_info_<a>-<b>.txt` —— 每 360 回合一个，per-episode 诊断
    - `warning.txt` —— PPO rollout 平均回报较上次跌幅 > 0.3 时追加警告
    - `pruning_search_log.txt` —— 主日志（修复后应该只有一个 banner，不是 80 个）
- **预期耗时**：6000 episodes × 59 steps，每 episode ~4-5 秒（含 ReplanSession 调用），
  全程 ~6-7 小时（参考之前 5000 episodes 跑了 ~6h08m）。

## 验证清单（跑起来后第一时间检查）

```bash
# launcher 启动时会写 LATEST_RUN_DIR 指针到算法分支根
RUN_DIR="$(cat 'Parting Chapter/persistent/rl/bert-base/mrpc/LATEST_RUN_DIR')"
LOG="$RUN_DIR/stage2_noise/pruning_search_log.txt"

# 1) 输出目录是否回到 Parting Chapter（而不是 Paean/outputs）
echo "RUN_DIR=$RUN_DIR"

# 2) pruning_search_log.txt 顶部不应该有 80 个重复 header
awk 'NR<=10' "$LOG"

# 3) details/ 目录应该开始有文件（前 360 回合后）
ls -lh "$RUN_DIR/stage2_noise/details/" 2>/dev/null

# 4) 框线 ╭─╮│╰╯ 不应该出现在新日志里
grep -c -e '╭' -e '╰' -e '│' "$LOG"
# 期望: 0

# 5) warning.txt 一开始应该不存在（仅 reward drop > 0.3 时才写）
ls -lh "$RUN_DIR/stage2_noise/warning.txt" 2>/dev/null || echo "warning.txt absent (期望)"
```

## 切换到其他常用任务时（备查，agent 不读这一段）

需要换任务时，**直接覆盖上面的 active command 代码块** + 改这里的元信息。下面只是常用命令样板，不会被执行：

- 续训（同 preset 不带 `--fresh`，自动检测持久化目录）：
  `bash llama_7B_LayerImportance.sh run rl --preset mrpc-blb-stage2-rl`
- Final eval（拿训练期 best action 跑完整 eval）：
  `bash Paean/run_final_eval.sh --preset mrpc-final-eval-only --action-config "$RUN_DIR/blb_stage2/progress/diagnostics/best_action_vec.json"`
- 多 seed 扫（5 seeds, 隔离持久化目录）：
  `bash tools/run_multi_seed.sh mrpc-blb-stage2-rl 1,2,3,4,5 trial1 --fresh`
- 旧 single-shot 路径回退：
  `bash llama_7B_LayerImportance.sh run rl --preset mrpc-blb-stage2-rl --fresh --blb-v3-no-sequential-rl`

## 服务器 agent 期望

- agent 只读这个文件的**第一个 ```bash 代码块**，其余 markdown 全部忽略。
- agent 应该在仓库根目录 `bash` 执行（不要 `cd`，所有路径已经按相对仓库根写好）。
- 如果该文件未变更（git hash 未动），agent 不应重复触发同一命令 —— 由 agent 侧做幂等。
