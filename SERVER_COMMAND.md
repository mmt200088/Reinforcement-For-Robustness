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
- **本次实验假设**：600 轮已证明 collapse 修复有效，但 entropy/clip_fraction 在 anchor 后过低，可能导致 1w 轮搜索过窄。因此本次在保持 safe-neighbor mask 的前提下，提高并拉长 entropy schedule，扩大 neighbor mutation/radius：
  - `--stage2-search-episodes 10000`
  - `--blb-v3-warmstart-anchor-episodes 120`
  - `--blb-v3-ent-coef 0.04`
  - `--blb-v3-ent-coef-ramp-episodes 1200`
  - `--blb-v3-warmstart-neighbor-ramp-episodes 3000`
  - `--blb-v3-warmstart-neighbor-max-mutations 16`
  - `--blb-v3-warmstart-neighbor-max-radius 3`
- **硬失败示例**：`loss_mean>=99`、NaN/inf、持续 P1(acc)、invalid steps 重新出现、PPO `n_samples != 60*59`、20 分钟无 episode 增长、双卡未使用。
- **软异常示例**：超过 2000 episodes 没有新 best、entropy 接近 0 且没有进展、clip_fraction 连续偏高、GPU1 active rate 过低。
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
