# 服务器端运行控制文件（SERVER_COMMAND.md）

> **协议**：服务器 agent 监听本文件 → 提取**第一个 ```bash 代码块** → 在仓库根目录 `bash` 执行。
> 本地这边改一次 + push，远端下次同步 / 触发就会按新命令跑。下方 metadata 段只是给人看的，agent 不解析。

## ▶ active command

```bash
set -euo pipefail
bash scripts/stage2_reward_probe_scaling_benchmark.sh
```

## metadata

- **任务**：新四卡 GPUShare 服务器迁移前，验证 Stage-2 RL reward probe 的 1/2/3/4 卡并行速度和 batch size。
- **协议**：服务器只负责 `git pull`、运行实验、产出结果 artifacts；真实源码修改都在本地完成并通过 git 同步。
- **本次目标**：
  - 使用真实 Stage-2 reward probe 链路测试 `CUDA_VISIBLE_DEVICES=0`、`0,1`、`0,1,2`、`0,1,2,3`。
  - 固定 `--stage2-k-trials 4` 和 `--stage2-probe-size 256`；四卡时每张卡正好跑一个独立噪声 trial。
  - 扫描 batch size `64 128 256`，选择 reward probe 平均 wall time 最低的配置。
  - 输出 `stage2_reward_probe_scaling_report.html`、`benchmark_summary.json`、每个配置的 `episodes.jsonl` 和日志。
  - 速度验证通过后，正式长 run 使用 `bash scripts/stage2_first10k_server_run.sh`，该脚本默认四卡、K=4、batch size=256。
- **本次实验假设**：上一版 Pareto-only 10k 全程 P3、无 invalid、无稳定性失败，但后期 frontier 扩展少，dominated/duplicate 样本占比高，且旧 MLP policy/critic 表达力不足、baseline prior 过容易把策略锁在 baseline 附近。Claude 已修复 action → 真实模型计算配置链路，旧结果只能作为 baseline。新 run 使用 GTrXL token policy、外部衰减 baseline prior、PPO 稳定器，以及 non-monotonic empirical proposal sampler。历史 raw radius2 曾在 `radius=2, mutations=8/9` 区间触发 P1 cluster，因此仍保持默认 raw `NEIGHBOR_MAX_RADIUS=1`，只在 frontier 停滞且最近健康时打开受控 radius2：
  - `--stage2-search-episodes 10000`
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
- **同步保护**：脚本中的 `git pull --ff-only` 如果失败或超时会直接 abort，不能继续用旧 HEAD 跑 10k。
- **Pareto cost 目标**：P1/P2 不入 cost archive；P3 候选只按 `fusion_gain / k_gain / bits_gain` 的非支配 frontier 给 PPO bounded shaping，不再用 `typical_*` 或人工权重标量决定 cost 排名。
- **Policy/critic 目标**：`BLBStage2SequentialPolicy` 应为 `blb_v3_sequential_gtrxl_v2scale`：causal GTrXL `d_model=256, n_heads=8, n_layers=4, d_ff=512, dropout=0.1`，per-slot heads，单 value head；旧 sequential checkpoint 不兼容，必须 fresh。
- **PPO 稳定器**：运行日志/`ppo_updates.jsonl` 应包含 approximate KL、KL early stop、adaptive LR scale、return normalizer、per-slot entropy recovery 等新字段。
- **Non-monotonic 探索目标**：不要把“降低 SF”当成必然靠近边界。SF/K 的 index move 只是 proposal；真实边界方向只能由 F1 model-forward metric/stability、Rescale_optimizer cost signals 和 Pareto archive 事件确认。允许某些 SF/K 反向或横向 move，如果它们满足 P3 并改善 `fusion_gain/k_gain/bits_gain`。
- **guarded radius2 规则**：只有 absolute episode 达到 1060、最近 600 轮 frontier expansion 少于 1、最近 100 轮无 P1/P2/invalid/stability/loss-cap，且 offset 在 radius1 历史中至少 3 次 P3 成功且 0 次失败，才允许 radius2；若 radius2 触发硬失败，立刻 300 episode cooldown 回到 radius1。
- **硬失败示例**：四卡配置没有出现 4 个 probe worker、`terminal_probe_trial_counts` 不是 `[1,1,1,1]`、GPU 采样显示部分卡完全空闲、四卡 wall time 没有明显优于单卡且不是 batch size/启动开销造成。
- **主要输出**：`experiments/server_command_runs/stage2_reward_probe_scaling_<timestamp>/`
  - `stage2_reward_probe_scaling_report.html`
  - `benchmark_summary.json`
  - `best_batch_size.txt`
  - `runs.jsonl`
  - `*_episodes.jsonl`
  - `*_pruning_search_log.txt`
  - `benchmark_stdout.log`
