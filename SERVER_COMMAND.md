# 服务器端运行控制文件（SERVER_COMMAND.md）

> **协议**：服务器 agent 监听本文件 → 提取**第一个 ```bash 代码块** → 在仓库根目录 `bash` 执行。
> 本地这边改一次 + push，远端下次同步 / 触发就会按新命令跑。下方 metadata 段只是给人看的，agent 不解析。

## ▶ active command

```bash
set -euo pipefail
PLANNED_EPISODES=60000 \
SMOKE_EPISODES=1000 \
BATCH_SIZE=512 \
K_TRIALS=4 \
FAST_REWARD_MODE_ENABLED=1 \
ONLINE_K_TRIALS=1 \
TERMINAL_EVAL_BATCH_SIZE=4 \
PROMOTION_VALIDATION_TRIALS=4 \
PROMOTION_MARGIN_WINDOW=0.25 \
PROBE_SIZE=256 \
RL_CUDA_VISIBLE_DEVICES=0,1,2,3 \
REWARD_DEVICES=0,1,2,3 \
MAX_POST_ANCHOR_P12_RATE=0.30 \
P12_RATE_MIN_POST_ANCHOR=100 \
bash scripts/stage2_first10k_server_run.sh
```

## metadata

- **任务**：在新四卡 GPUShare 服务器上运行最新 Stage-2 BLB RL，正式目标 60000 轮，并统计每小时能跑多少轮。
- **协议**：服务器只负责 `git pull`、运行实验、产出结果 artifacts；真实源码修改都在本地完成并通过 git 同步。
- **本次目标**：
  - 先跑 1000 episode smoke；smoke PASS 后启动 fresh 60000 episode formal RL。
  - 使用最新 GTrXL Stage-2 sequential policy、adaptive scalar P3 cost reward、non-monotonic empirical proposal sampler、guarded radius2。
  - 默认高保真配置固定 `--stage2-k-trials 4` 和 `--stage2-probe-size 256`；四卡时每张卡跑一个独立噪声 trial。若本地已验证低 probe-size 训练加速配置，可以只在训练搜索阶段降低 `PROBE_SIZE`，最终候选仍要回到 256 probe 做验证。
  - 使用上一轮四卡 benchmark 选出的最快配置：`batch size=512`、`reward_devices=0,1,2,3`。
  - 本轮启用 fast online reward mode：正式训练在线 reward 用 `ONLINE_K_TRIALS=1`，每批最多 4 个 terminal action 通过 `ProbeRunner.run_action_trials_once(...)` 分配到四张卡，每张卡评估一个不同 action；优秀/边界 P3 候选用 `PROMOTION_VALIDATION_TRIALS=4` 走重复 trial 复验。基线/noisy preflight 和最终 promotable 证据仍保留 K=4 语义。
  - 监控吞吐：至少在运行约 1 小时后统计 episode/hour；后续继续从 `episodes.jsonl` 和 `monitor_live.json` 更新速度。
  - 本轮 P1/P2 判断按用户新标准：post-anchor P1+P2 比例不超过 30% 不停跑；少量 P1/P2 只作为 warning。invalid steps、loss-cap burst、非有限 PPO、四卡 reward-probe 失效仍然是硬失败。
  - 遇到硬失败或明显训练异常时及时停止，不浪费服务器时间；代码 bug 必须回本地修复后再 push、server pull、rerun。
- **本次实验假设**：上一版 Pareto-only 10k 全程 P3、无 invalid、无稳定性失败，但后期 frontier 扩展少，dominated/duplicate 样本占比高，reward 对“可学习的 cost 梯度”不够敏感。当前 run 回到 adaptive scalar P3 cost：fusion_count 和 truncation/K gain 有清晰区间式 reward jump，total_bits 只作为弱线性 tie-breaker；P1/P2 仍完全不能被 cost 抵消。新 run 使用 GTrXL token policy、外部衰减 baseline prior、PPO 稳定器，以及 non-monotonic empirical proposal sampler。历史 raw radius2 曾在 `radius=2, mutations=8/9` 区间触发 P1 cluster，因此仍保持默认 raw `NEIGHBOR_MAX_RADIUS=1`，只在 frontier 停滞且最近健康时打开受控 radius2：
  - `--stage2-search-episodes 60000`
  - `--blb-v3-warmstart-anchor-episodes 60`
  - `--blb-v3-ent-coef 0.06`
  - `--blb-v3-ent-coef-ramp-episodes 600`
  - `--blb-v3-warmstart-bias-gain 1.2`
  - `--blb-v3-warmstart-neighbor-ramp-episodes 1800`
  - `--blb-v3-warmstart-neighbor-max-mutations 12`
  - `--blb-v3-warmstart-neighbor-max-radius 1`
  - `--blb-v3-guarded-radius2-enabled 1`
  - `--blb-v3-guarded-radius2-min-episode 1060`
  - `--blb-v3-guarded-radius2-stall-window 600`
  - `--blb-v3-guarded-radius2-max-mutations 4`
  - `--blb-v3-guarded-radius2-episode-fraction 0.15`
  - `--blb-v3-guarded-radius2-cooldown-episodes 300`
- **长周期判断**：历史经验上，Stage-2 RL 通常需要 50000+ 轮才有有效搜索结论；但如果到 20000 轮后 reward 仍长期没有进入快速增长期，要把它当成需要诊断的搜索/训练异常，而不是简单继续耗时。
- **同步保护**：脚本中的 `git pull --ff-only` 如果失败或超时会直接 abort，不能继续用旧 HEAD 跑 60000。
- **Budgeted adaptive scalar cost 目标**：P1/P2 不吃任何 cost reward；P3 内部将 metric margin 和 cost 分开预算，metric margin 只占小预算，不能挤掉 cost ranking。P3 候选中 `fusion_gain` 每 +1 给明显区间式 boost，truncation/K gain 每跨一个 layer-equivalent average-K tier 给同等级 boost，当前默认 `cost_k_step_size=1/12`。这不是拍脑袋参数：2026-05-23 用 fast-reward 真实 episodes 离线 sweep 后发现旧 `1/59` single-slot K tier 会让约 27.5% P3 候选过早打满 P3 cost clip，`1/12` 把饱和率降到约 9%，同时 fusion/K 仍有清晰阶梯；`total_bits` 只给单独 clip 的弱线性 tie-breaker，不能接近一个 fusion/K tier step。`ParetoCostArchive` 仍可记录 P3 frontier，用于诊断和 empirical exploration 统计，但不作为默认 PPO scalar reward。
- **Static invalid-level pre-mask**：训练开始前会做一次 baseline-prefix one-slot Rescale_optimizer 可行性扫描，参考 COINN 先缩小 invalid 配置空间再优化的思想。扫描只调用 `evaluate_step`，最后一步不 `commit_step`，因此不会触发 terminal model-forward；被局部判为 invalid 的 `(layer, block, slot, level)` 会在 PPO 采样前从 `action_level_mask` 隐藏。它比 runtime empirical mask 更 aggressive，允许牺牲一部分可能在其他 prefix 下才合法的组合，以减少 invalid-chain 替换和优化器重试。
- **Policy/critic 目标**：`BLBStage2SequentialPolicy` 应为 `blb_v3_sequential_gtrxl_v2scale`：causal GTrXL `d_model=256, n_heads=8, n_layers=4, d_ff=512, dropout=0.1`，per-slot heads，单 value head；旧 sequential checkpoint 不兼容，必须 fresh。
- **PPO 稳定器**：运行日志/`ppo_updates.jsonl` 应包含 approximate KL、KL early stop、adaptive LR scale、return normalizer、per-slot entropy recovery 等新字段。
- **Non-monotonic 探索目标**：不要把“降低 SF”当成必然靠近边界。SF/K 的 index move 只是 proposal；真实边界方向只能由 F1 model-forward metric/stability、Rescale_optimizer cost signals 和 P3 adaptive scalar cost/diagnostic archive 确认。允许某些 SF/K 反向或横向 move，如果它们满足 P3 并改善 `fusion_gain/k_gain/bits_gain`。
- **guarded radius2 规则**：只有 absolute episode 达到 1060、最近 600 轮 frontier expansion 少于 1、最近 100 轮无 P1/P2/invalid/stability/loss-cap，且 offset 在 radius1 历史中至少 3 次 P3 成功且 0 次失败，才允许 radius2；若 radius2 触发硬失败，立刻 300 episode cooldown 回到 radius1。
- **硬失败示例**：四卡配置没有出现 4 个 probe worker、`terminal_probe_trial_counts` 不是 `[1,1,1,1]`、GPU 采样显示部分卡完全空闲、四卡 wall time 没有明显优于单卡且不是 batch size/启动开销造成。
- **四卡验证**：final monitor 必须检查 `terminal_probe_devices` 包含 `cuda:0..cuda:3`、`terminal_probe_trial_counts` 为 `[1,1,1,1]`，并且 `nvidia_smi_during_rl.csv` 中四张卡都出现非零利用率；不能只看 GPU0/1。
- **主要输出**：`experiments/server_command_runs/stage2_rl_60000_curve_<timestamp>/`
  - `server_command_stdout.log`
  - `run_manifest.json`
  - `smoke_monitor_summary.json`
  - `formal_monitor_summary.json`
  - `formal_episodes.jsonl`
  - `ppo_updates.jsonl`
  - `pareto_frontier.json`
  - `nvidia_smi_during_rl.csv`
  - `server_monitor_report.html`
