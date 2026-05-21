# 服务器端运行控制文件（SERVER_COMMAND.md）

> **协议**：服务器 agent 监听本文件 → 提取**第一个 ```bash 代码块** → 在仓库根目录 `bash` 执行。
> 本地这边改一次 + push，远端下次同步 / 触发就会按新命令跑。下方 metadata 段只是给人看的，agent 不解析。

## ▶ active command

```bash
set -euo pipefail
bash scripts/stage2_first10k_server_run.sh
```

## metadata

- **任务**：停止旧 MLP sequential run 后，使用本地实现的 v2-scale GTrXL policy/critic 和 non-monotonic cost-boundary exploration 重新 fresh 运行 BLB Stage-2 sequential RL。
- **协议**：服务器只负责 `git pull`、运行实验、产出结果 artifacts；真实源码修改都在本地完成并通过 git 同步。
- **本次目标**：
  - 运行 server-side sequential smoke + BLB contract tests + `blb_verify_noise_install` 链路检查。
  - 先启动 fresh 1000 episode dual-GPU smoke run，确认新 action → cfg 链路、GTrXL policy、PPO stabilizers、non-monotonic exploration 和 guarded radius2 诊断正常。
  - smoke PASS 后再启动 fresh 10k dual-GPU Stage-2 sequential RL run，`--skip-final-eval`，避免训练完成后被 final eval 拖住。
  - 使用两张 GPU 并行 reward probe：`CUDA_VISIBLE_DEVICES=0,1` + `--blb-v3-reward-devices 0,1`。
  - 在线 watchdog 每分钟读取 structured `episodes.jsonl` / `ppo_updates.jsonl` / GPU 采样，发现硬失败就优雅停止并保留 partial artifacts。
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
- **硬失败示例**：NaN/inf、持续或高频 P1(acc)、重复 `loss_mean>=99`、invalid steps 重新出现、PPO `n_samples != 60*59`、20 分钟无 episode 增长、双卡未使用。偶发负 reward 尖刺或孤立 P1 不单独判失败，关键看 rolling mean 和异常频率。
- **软异常示例**：孤立 `loss_mean>=99`、超过 2000 episodes 没有新 best、entropy 接近 0 且没有进展、clip_fraction 连续偏高、GPU1 active rate 过低。
- **主要输出**：`experiments/server_command_runs/stage2_rl_first10k_curve_<timestamp>/`
  - `monitor_live.json`
  - `monitor_events.jsonl`
  - `monitor_summary.json`
  - `server_monitor_report.html`
  - `reward_windows.csv`
  - `episode_health_windows.csv`
  - `smoke_monitor_summary.json`
  - `smoke_server_monitor_report.html`
  - `formal_monitor_summary.json`
  - `formal_server_monitor_report.html`
  - `episodes.jsonl`
  - `ppo_updates.jsonl`
  - `nvidia_smi_during_rl.csv`
  - `rl_10000_dual_gpu.log`
