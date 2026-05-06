# BLB Stage-2 RL 使用说明

本文档只面向运行与调参。当前项目在服务器上统一通过
`bash llama_7B_LayerImportance.sh ...` 启动；不要直接调用底层训练入口。

更完整的训练逻辑、目录结构、优雅停止与续训练说明见
[`docs/BLB_stage2_rl_FULL_FLOW.md`](BLB_stage2_rl_FULL_FLOW.md)。

## 一句话概览

`blb_v3` 是当前默认的 Stage-2 RL 实现。它不再只搜索旧版
`*_scaling_factors`，而是面向 BLB 的 5 个 block 和 first-input fresh 噪声候选点，
把动作解释成一份 BLB 噪声配置，再根据精度、稳定性和 CKKS/MPC 成本信号训练 PPO。

旧版实现仍保留为 `legacy_v2`，用于复现实验或回退。两条路径通过
`--stage2-rl-variant` 切换，持久化 checkpoint 文件名也不同，因此不会互相覆盖。

## 推荐启动方式

已经新增 preset：

```bash
bash llama_7B_LayerImportance.sh run rl --preset mrpc-blb-stage2-rl --fresh
```

同一参数组合续训练时去掉 `--fresh`：

```bash
bash llama_7B_LayerImportance.sh run rl --preset mrpc-blb-stage2-rl
```

只跑一次 BLB 最高配置 final eval：

```bash
bash llama_7B_LayerImportance.sh eval --preset mrpc-blb-max-final-eval
```

这个 preset 使用 `--final-eval-source blb-max`。含义是：Stage-1 的 GELU/Softmax 从
`glue_final_configs_best_ppo.json` 中读取 MRPC 推荐配置；Stage-2 不读取搜索结果，也不要求手写
`--manual-stage2-noise`，而是直接选择当前 final-eval 路径可执行的最大动作/最大 scaling-factor 配置。
random 对照组数量默认为 0，只保留 Baseline、Optimized/max 与 Stage1Fixed+MaxSF 等核心组别，便于先看最高配置效果。

脚本会自动 `nohup` 后台运行，并在启动后打印 PID、日志路径和持久化目录。
常见查看方式：

```bash
tail -f rl_results/persistent/rl/bert-base/mrpc/s1t0.005_s2t0.005_s2st0.005/logs/blb_stage2_rl.log
```

## BLB 与旧版切换

默认就是 BLB：

```bash
bash llama_7B_LayerImportance.sh run rl --dataset mrpc --episodes 51000,80000 --fresh
```

显式指定 BLB：

```bash
bash llama_7B_LayerImportance.sh run rl --dataset mrpc --stage2-rl-variant blb_v3 --fresh
```

切回旧版 Stage-2 RL：

```bash
bash llama_7B_LayerImportance.sh run rl --dataset mrpc --stage2-rl-variant legacy_v2 --fresh
```

`legacy_v2` 下不要再传 BLB 专属参数；launcher 会在启动前直接报错，避免把两套实现混在一起。

## 命令行参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--stage2-rl-variant blb_v3|legacy_v2` | `blb_v3` | 选择 Stage-2 RL 实现。`blb_v3` 是 BLB 新实现，`legacy_v2` 是旧噪声 scaling-factor PPO |
| `--ppo-update-interval N` | `120` | 旧版 PPO 更新间隔；在 BLB 下也会作为默认 rollout size |
| `--stage2-rollout-size N` | 跟随 `--ppo-update-interval` | BLB PPO 每收集多少 episode 做一次 update；需要单独调 BLB 时用它覆盖 |
| `--blb-v3-rollout-size N` | 同上 | `--stage2-rollout-size` 的长别名 |
| `--stage2-save-interval N` | `200` | BLB live checkpoint 保存间隔 |
| `--blb-v3-save-interval N` | `200` | `--stage2-save-interval` 的长别名 |
| `--stage2-eval-interval N` | `100` | BLB 训练日志评估间隔，只用于观察策略状态 |
| `--blb-v3-eval-interval N` | `100` | `--stage2-eval-interval` 的长别名 |
| BLB v3 Rescale_optimizer | `Rescale_optimizer` | 固定使用真实 in-process `Rescale_optimizer`；不再提供 heuristic/stub/subprocess 命令行选择 |
| `--stage2-calibrate-baseline-samples N` | `8` | 训练前用多少随机动作校准 reward 权重 |
| `--blb-v3-calibrate-baseline-samples N` | `8` | `--stage2-calibrate-baseline-samples` 的长别名 |

BLB 仍然复用现有 Stage-2 外壳参数：

| 参数 | 说明 |
| --- | --- |
| `--stage2-search-episodes N` 或 `--episodes S1,S2` | Stage-2 总训练 episode 数 |
| `--stage2-search-lr FLOAT` | Stage-2 PPO 学习率 |
| `--stage2-limit-tolerance FLOAT` | Stage-2 指标容忍比例 |
| `--stage2-stability-tolerance FLOAT` | Stage-2 稳定性容忍比例 |
| `--stage2-k-trials INT` | 每个候选动作做多少次噪声 trial |
| `--stage2-probe-size INT` | 稳定性评估使用的固定探针子集大小 |
| `--stage2-fixed-config-source stage1_result|json|manual` | Stage-2 固定 GELU/Softmax 的来源 |
| `--stage2-fixed-config PATH` | `json` 来源下的合并配置文件 |
| `--final-eval-source max|stage2-max|blb-max` | final-eval-only 时直接选择 Stage-2 最大动作/最大 scaling-factor；Stage-1 仍从 JSON/search/manual 回退 |

## 产物目录

BLB checkpoint 存在当前 run 的 Stage-2 进度目录：

```text
<run_dir>/stage2_noise/progress/
├── blb_stage2_rl_checkpoint_live.pt
├── blb_stage2_rl_checkpoint_final.pt
├── blb_stage2_best_cfg.pkl
└── STOP_RL
```

其中 `STOP_RL` 不是默认产物，而是需要优雅停止时手动创建的标志文件。

旧版 Stage-2 RL 的 checkpoint 仍是 `noise_rl_checkpoint.pt`。resume 逻辑会根据
`--stage2-rl-variant` 自动选择对应文件，不会把 BLB checkpoint 当成旧版 checkpoint 加载。

## 优雅停止与续训练

优先用启动脚本打印的 PID：

```bash
kill -INT <PID>
```

也可以在 BLB 进度目录创建停止标志：

```bash
touch <run_dir>/stage2_noise/progress/STOP_RL
```

BLB 会在当前 episode 结束、并且到达安全保存点后写出 live checkpoint，然后退出。
续训练不需要传 `--resume-from`；再次运行同一条 `bash llama_7B_LayerImportance.sh ...`
命令即可。launcher 会根据相同的数据集、模型和约束参数定位同一个持久化目录，并把它交给训练流程恢复。

## Rescale Optimizer

BLB Stage-2 RL 训练固定使用真实 `Rescale_optimizer` in-process 路径，不再提供
`heuristic`、`stub` 或 `subprocess` 选择。CKKS 模数链、fusion 和 total_bits 必须由
`Rescale_optimizer` 算法计算；如果 `Rescale_optimizer/configs/{dataset}` 或
`static_skeletons_{dataset}.json` 无法加载，训练会直接报错并停止。
