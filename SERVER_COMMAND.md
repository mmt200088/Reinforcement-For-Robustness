# 服务器端运行控制文件（SERVER_COMMAND.md）

> **协议**：服务器 agent 监听本文件 → 提取**第一个 ```bash 代码块** → 在仓库根目录 `bash` 执行。
> 本地这边改一次 + push，远端下次同步 / 触发就会按新命令跑。下方 metadata 段只是给人看的，agent 不解析。

## ▶ active command

```bash
set -e

# ----------------------------------------------------------------------
# 诊断当前 RL-best ep203 的 8 个 invalid_chain 究竟落在哪些 (layer, block)
# 上，以及优化器给出的具体 reason。脚本是纯诊断（torch + Rescale_optimizer
# 都需要，所以必须在跑训练的 conda 环境里执行）。
#
# 输出：
#   reports/blb_opt/invalid_blocks/rl_best_ep203_buggy_reward/{report.md,report.json}
# 每行一条记录：(L, B) · graph_key · 失败原因 · 该 block 内每个 slot 的 SF/K 决策。
# 如果你想再对 baseline action 跑同样的诊断（应该是 0 invalid，做 sanity），
# 把下面的 ACTION_JSON 改成 baseline_action_vec.json 路径再触发一次即可。
# ----------------------------------------------------------------------

ACTION_JSON="Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.005_s2t0.005_s2st0.005/stage2_noise/progress/diagnostics/best_action_vec.json"
OUT_DIR="reports/blb_opt/invalid_blocks/rl_best_ep203_buggy_reward"

if [ ! -f "$ACTION_JSON" ]; then
  echo "[FATAL] action JSON not found: $ACTION_JSON" >&2
  exit 1
fi

echo "================================================================================"
echo "BLB invalid-chain diagnosis · action=$ACTION_JSON"
echo "================================================================================"

python scripts/blb_diagnose_invalid_blocks.py \
    --action-config "$ACTION_JSON" \
    --profile mrpc \
    --num-layers 12 \
    --rescale-optimizer-root Rescale_optimizer \
    --output-dir "$OUT_DIR"

echo ""
echo "================================================================================"
echo "DONE · 报告已写入 $OUT_DIR/{report.md, report.json}"
echo "本地 pull 后看 reports/blb_opt/invalid_blocks/rl_best_ep203_buggy_reward/report.md"
echo "================================================================================"
```

---

## 元信息（meta，给人看的，agent 忽略）

- **任务**：跑 `scripts/blb_diagnose_invalid_blocks.py` 把 RL-best ep203（buggy reward 时期的训练 best）送进 Rescale_optimizer，逐 (layer, block) 列出哪 8 个失败、失败原因是什么。
- **更新时间**：2026-05-17 凌晨 03:30+
- **背景**：
  - 上次 final-eval 结果：baseline action（全 max SF）→ 59 valid / 0 invalid（Acc=0.879），RL-best ep203 → 51 valid / 8 invalid（Acc=0.316, F1=0.152, NaN loss）。
  - **8 个 invalid 块就是 RL-best 表现崩溃的根本原因** —— 装上之后这些 (layer, block) 模数链都失败，BLB 噪声无法装进模型，prediction 退化到接近随机。
  - final-eval 的聚合 JSON 只记 `invalid_count=8`，**不写具体是哪 8 个块**。所以需要这个 sidecar 把 8 个块 + 各自失败原因捞出来。

### 为什么会有 invalid_chain？是不是 bug？

短答：**不是 bug**，是 RL 选了优化器无法满足的 SF/K 组合，优化器尽职报告 `invalid_chain`。

证据链：

| 现象 | 结论 |
|------|------|
| baseline action（全 max SF）→ 0 / 59 invalid | 接口对 baseline 是正确的，否则 baseline 不可能 100% valid。 |
| RL random-ish 动作（broken reward）→ ~10–15% slot invalid | 动作空间包含 infeasible 区域，每次随机命中就触发 `invalid_chain`。 |
| `first_invalid_counts.json` 显示 L01-B1: 973 / L05-B5: 955 集中爆 | 某些 (layer, block) 模数链格外紧 —— 这两个块的 SF 一旦砍掉档位就会破坏 prime 上界 / fusion 减少不足，是数学事实而非 bug。 |
| 优化器返回的 `invalid_chain` 带 `reason="new chain has prime(s) > q_max=60 at stage(s) [1]; fusion cannot reduce. Reject."` | 这是 Rescale_optimizer 在尽职说"该动作的模数链超过 q_max 上界，没法 fuse 修"。属于优化器的正常拒绝逻辑。 |

修复路径：
1. **Reward 必须能差分** —— 这就是上次 commit `173596d` 修的"noisy baseline preflight"。修完后 priority 1/2 阈值合理，PPO 才能感知到"选大 SF 比小 SF 更优"。
2. **Invalid 步数自带 penalty** —— sequential reward 里 `invalid_penalty=1.0` 已经在惩罚每个 invalid sub-step；PPO 在修好 terminal reward 后会学会避开。
3. **Stage 2 → 把所有 invalid 块都写进 details/，不只是 first**（本次 commit）—— 这样训练时 grep details 文件就能看到每个 episode 在哪些 (layer, block) 翻车，方便定位"特别难"的块。

### 这次脚本会做什么

| 步骤 | 命令 | 说明 |
|------|------|------|
| 1 | `python scripts/blb_diagnose_invalid_blocks.py --action-config best.json` | 解析 best_action_vec.json → 装 `RescaleOptimizerBridge` (InProcessInvoker) → 调 `evaluate_action_for_cost` → 拿到 59 个 per-config 输出 → 逐条打印 valid/invalid 和原因 |
| 2 | `--output-dir reports/blb_opt/invalid_blocks/rl_best_ep203_buggy_reward/` | 同时落两份文件：`report.md`（人读） + `report.json`（机器读） |

### 预期产物

`reports/blb_opt/invalid_blocks/rl_best_ep203_buggy_reward/report.md`：
- 顶部 summary：n_total / n_valid / n_invalid / total_bits / fusion / avg_k
- "Invalid blocks" 表：`(L, B) · graph_key · total_bits · fusion · reason`
- "Slot configs of invalid blocks"：每个失败块下面列出该块的 12 个左右 slot 的 SF / K 选择，便于直接看到"哪个槽位选了什么导致了不可行"

`report.json`：同样数据机器格式，便于后续脚本聚合。

### 预期耗时

- 加载 BLB 包 + Rescale_optimizer 预加载 baselines：~10s
- 跑 59 次 `replan_with_user_actions`：~10s（in-process invoker，每次 ms 级）
- 写文件：<1s
- **总计 ≤ 30 秒**

## 之后该做什么（不在本脚本范围内，写给操作者参考）

1. **再开始一轮 RL**（修好 reward + 新增 invalid 可见性后）：
   ```bash
   bash llama_7B_LayerImportance.sh run rl --preset mrpc-blb-stage2-rl --fresh
   ```
   修复后的 reward 会给 invalid 步 -1.0 罚 + acc/stab/cost 三层差分信号，PPO 应该几百 episode 就能学会避开 L01-B1 / L05-B5 这种"难"块。
2. **如果还是大量 invalid**，pull 回 `details/noise_ppo_step_info_*.txt`，每个 episode 现在会逐条列出 "invalid_blocks: L00-B3 graph=block3_exp_n4 reason=...". 直接 grep 哪个 (L, B) 频次最高、哪个 reason 重复出现，再回头看 action_space 是不是给了无效的 level（例如某个 SF 档位下限太低）。

## 切换到其他常用任务时（备查，agent 不读这一段）

需要换任务时，**直接覆盖上面的 active command 代码块** + 改这里的元信息。下面只是常用命令样板，不会被执行：

- 重新跑 RL（修复后版本，6000 episodes）：
  `bash llama_7B_LayerImportance.sh run rl --preset mrpc-blb-stage2-rl --fresh`
- 续训（同 preset 不带 `--fresh`，自动检测持久化目录）：
  `bash llama_7B_LayerImportance.sh run rl --preset mrpc-blb-stage2-rl`
- 重跑 final-eval（最新 best）：
  `bash Paean/run_final_eval.sh --preset mrpc-final-eval-only --action-config "$RUN_DIR/stage2_noise/progress/diagnostics/best_action_vec.json"`
- 多 seed 扫（5 seeds，隔离持久化目录）：
  `bash tools/run_multi_seed.sh mrpc-blb-stage2-rl 1,2,3,4,5 trial1 --fresh`
- 旧 single-shot 路径回退：
  `bash llama_7B_LayerImportance.sh run rl --preset mrpc-blb-stage2-rl --fresh --blb-v3-no-sequential-rl`

## 服务器 agent 期望

- agent 只读这个文件的**第一个 ```bash 代码块**，其余 markdown 全部忽略。
- agent 应该在仓库根目录 `bash` 执行（不要 `cd`，所有路径已经按相对仓库根写好）。
- 如果该文件未变更（git hash 未动），agent 不应重复触发同一命令 —— 由 agent 侧做幂等。
- 本次命令是**纯诊断**，不启动训练、不杀进程。可以随时跑，不会影响正在跑的 RL（如果有）。
