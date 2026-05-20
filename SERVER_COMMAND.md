# 服务器端运行控制文件（SERVER_COMMAND.md）

> **协议**：服务器 agent 监听本文件 → 提取**第一个 ```bash 代码块** → 在仓库根目录 `bash` 执行。
> 本地这边改一次 + push，远端下次同步 / 触发就会按新命令跑。下方 metadata 段只是给人看的，agent 不解析。

## ▶ active command

```bash
set -euo pipefail
bash scripts/stage2_first10k_server_run.sh
```

## metadata

- **任务**：优化 BLB Stage-2 sequential RL 前 10,000 episode 的 reward 曲线，而不是只验证 600 episode 不崩。
- **协议**：服务器只负责 `git pull`、运行实验、产出结果 artifacts；真实源码修改都在本地完成并通过 git 同步。
- **本次目标**：
  - 运行 server-side sequential smoke + BLB contract tests。
  - 启动 fresh 10k dual-GPU Stage-2 sequential RL run，`--skip-final-eval`，避免训练完成后被 final eval 拖住。
  - 使用两张 GPU 并行 reward probe：`CUDA_VISIBLE_DEVICES=0,1` + `--blb-v3-reward-devices 0,1`。
  - 在线 watchdog 每分钟读取 structured `episodes.jsonl` / `ppo_updates.jsonl` / GPU 采样，发现硬失败就优雅停止并保留 partial artifacts。
- **本次实验假设**：600 轮已证明 collapse 修复有效。第一次 10k 尝试在 `NEIGHBOR_RAMP=3000, max_mutations=16, max_radius=3` 下到 1784 轮被 watchdog 停止；P1 集中出现在 `radius=2, mutations=8/9`，而 `radius=1` 到约 1500 轮没有 P1 且 reward 均值继续上升。因此本次保留 entropy schedule，但把 safe-neighbor 放宽速度降下来，只允许 radius=1：
  - `--stage2-search-episodes 10000`
  - `--blb-v3-warmstart-anchor-episodes 120`
  - `--blb-v3-ent-coef 0.04`
  - `--blb-v3-ent-coef-ramp-episodes 1200`
  - `--blb-v3-warmstart-neighbor-ramp-episodes 6000`
  - `--blb-v3-warmstart-neighbor-max-mutations 8`
  - `--blb-v3-warmstart-neighbor-max-radius 1`
- **硬失败示例**：NaN/inf、持续或高频 P1(acc)、重复 `loss_mean>=99`、invalid steps 重新出现、PPO `n_samples != 60*59`、20 分钟无 episode 增长、双卡未使用。偶发负 reward 尖刺或孤立 P1 不单独判失败，关键看 rolling mean 和异常频率。
- **软异常示例**：孤立 `loss_mean>=99`、超过 2000 episodes 没有新 best、entropy 接近 0 且没有进展、clip_fraction 连续偏高、GPU1 active rate 过低。
- **主要输出**：`experiments/server_command_runs/stage2_rl_first10k_curve_<timestamp>/`
  - `monitor_live.json`
  - `monitor_events.jsonl`
  - `monitor_summary.json`
  - `server_monitor_report.html`
  - `reward_windows.csv`
  - `episode_health_windows.csv`
  - `episodes.jsonl`
  - `ppo_updates.jsonl`
  - `nvidia_smi_during_rl.csv`
  - `rl_10000_dual_gpu.log`
