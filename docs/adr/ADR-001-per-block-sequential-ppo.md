# ADR-001: Per-block Sequential PPO（取代 single-shot 577-dim）

- **Status**: Accepted
- **Date**: 2026-05-15
- **Tags**: rl-design, policy, action-space

## Context

BLB Stage-2 RL 的动作空间长度为 **577 dim**（L=12 时：12 层 × 48 槽 +
1 first_input），每个槽位 4-6 个候选 → 名义搜索空间约 5^577。

原 single-shot 设计（`BLBStage2Env` / `BLBStage2Policy`）一个 episode 跑
一次 `env.step(action_vec)`，让 PPO 在一次 reward 信号下同时决定全部 577
维。冷启动训练观察到：
- 大量 episode 是 invalid（policy 随机采样几乎必踩坑）
- 即使 reward 来了，GAE 退化成 `A = r − V(s)`（horizon=1），没有时序
  credit assignment
- 训练 6000 episode 后 best_reward 仍剧烈震荡，靠近 baseline

数学上：用单步 reward 训 PPO 等价于 contextual bandit；577-dim 离散
action 的 bandit 在 RL 文献里基本无法收敛。

## Decision

把 episode 分解成 **horizon=59 的 sequential 决策**：
- step 0..3: layer 0 的 block 2, 3, 4, 5（layer 0 没有 block 1）
- step 4..58: layer 1..11 各 5 个 block
- first_input fresh slot 折进 step 0 一起出

每步只决策一个 (layer, block) 的子动作（7-12 维 MultiDiscrete），用
`ReplanSession` 对该 block 单独评估得到 **per-step dense cost reward**；
末步把累积的完整 vec 交给 `BLBStage2Env.step` 跑完整 model forward 得
hard-priority terminal reward。

实现位于 `blb_stage2_rl/sequential_env.py`、`sequential_policy.py`、
`sequential_runner.py`。`--blb-v3-sequential-rl true`（默认）启用；可以
`--blb-v3-no-sequential-rl` 显式回退老路径。

## Alternatives considered

| Option | Why rejected |
|--------|--------------|
| 继续 single-shot（horizon=1） | 不收敛（见 Context） |
| 按 **layer** 分（horizon=12） | 每步还要决策 ~48 dim；branching factor 仍偏大 |
| 按 **slot** 分（horizon=577） | episode 太长（59 已经够长了），GAE 截断与方差爆炸 |
| 直接上 hierarchical RL | 工程量太大；先看 horizon=59 能不能 work，能 work 就不上 |

## Consequences

**Positive**：
- 每步都有 dense cost reward，invalid 信号能在 5-10 episode 内被 PPO 学到
- 共享 trunk + 单一 MultiDiscrete head（13×6 = 78 logits）+ per-step
  slot mask，参数量比"每个 block 一个 head"小一个量级
- horizon=59 让 GAE-λ 真的能 carry value 信号

**Negative / trade-offs**：
- 每个 episode 需要 59 次 `ReplanSession` call（之前只需 1 次完整 model
  forward）。MRPC 实测单 episode wall-clock 慢 ~30%
- per-step reward shaping 系数（`invalid_penalty=1.0`,
  `cost_shaping_coeff=0.05`）需要手调；不慎会和 terminal hard-priority
  reward 量级冲突
- 老的 single-shot tests 还在跑（`BLBStage2Env` 没删），有 2 套测试要维护

**Things to watch**：
- 如果某天发现 per-step ReplanSession 是 bottleneck → 考虑批量化或
  cache（同样的 cfg 不重复 replan）
- 如果换成 GPT-2 / LLaMA 后 horizon 暴涨（更多 layer），考虑 hierarchical

## References

- Code: `blb_stage2_rl/sequential_env.py`, `sequential_policy.py`,
  `sequential_runner.py`
- Schedule helpers: `blb_stage2_rl/action_space.py::step_schedule`
- CLAUDE.md "Two episode formulations" 段
- Commit `d8efed0` (Add per-block sequential RL as the default BLB Stage-2 path)
