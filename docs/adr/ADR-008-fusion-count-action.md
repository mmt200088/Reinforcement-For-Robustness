# ADR-008: Per-block fusion-count action (vs per-slot scaling-factor decisions)

- **Status**: Accepted
- **Date**: 2026-06-03
- **Deciders**: project owner + Claude
- **Tags**: action-space, stage2-rl, search-space

## Context

- 现状（before）：Stage-2 sequential RL 每个 `(layer, block)` 步用 GTrXL 的
  `max_step_dim≈24` 个 per-slot 头各选一个 scaling-factor 档；整体动作空间 ≈ `5^577`。
  为在这个巨大空间里 cold-start，代码堆了一整套机制：forced-baseline anchor、
  decaying warmstart prior、safe-neighbor mutation、`GuardedRadius2Controller`、
  三套 invalid-level mask（Static/Forbidden/Empirical）。
- 触发事件 / pain point：搜索空间太大、收敛慢、机制复杂且脆弱。用户要求把动作从
  「逐槽选 SF」改成「逐 block 选 fusion-count」，用一张离线映射表把巨大的 SF 组合空间
  压成「每个 fusion-count 下噪声最小的那几个 SF 组合里选一个」。
- 已知约束（不能违反的）：① 每个 reward 必须过真实 `Rescale_optimizer.replan`
  （CLAUDE.md critical model）；② RL 不决定 must-exist 操作是否发生（item 2）——rescale
  点是固定的，是否融合由优化器决定；③ baseline 选取逻辑不变（static_skeletons all-max）；
  ④ 不破坏 single-shot / F0 / candidate-store / 现有测试。

## Decision

每个 block 的 RL 决策从「全部 effective SF 槽」改成 `(fusion_option, K)` 两个 categorical。
`fusion_option` 由一张**离线预计算的 fusion-count 映射表**展开成「该 block 内全部 effective
SF 槽的具体 SF 动作」，再走完全不变的 `action_vector_to_cfgs → bridge → replan → reward`。

映射表构建（per block-type，7 种；block3 冻结不建）：枚举 effective chain 槽（rescale 只枚举
SF 值、**不枚举 None**，依据 item 2）→ 真实 replan → 按 realized `fusion_count` 分组 → 每组取
**post-override 实际安装方差**最小集（可插拔 `NoiseOrder`）→ 排序使 all-max baseline 落 option 0。

运行期：同一个 `BLBStage2SequentialPolicy` 用 `max_step_dim=2` 实例化；fusion 分支停用
safe-neighbor / radius2 / invalid-mask（map 全 valid），保留 anchor / warmstart / 熵 / KL-LR。
opt-in flag `--blb-v3-fusion-count-action`，与 substage 互斥。

**Why this choice**：搜索空间从 `5^577` 降到「每步 几~几十 option × 6 档 K」，且 fusion 主要是
「关掉」复杂机制 + 复用现有 PPO/持久化/诊断/四卡脚手架，改动面比重写小得多。

## Alternatives considered

| Option | Why rejected |
|--------|--------------|
| 保留逐槽 SF（现状） | 空间 `5^577`、cold-start 难、机制复杂——正是要解决的问题。|
| 枚举也含 rescale-None（让 RL 主动「丢」rescale） | 违反 item 2（RL 不决定操作存在与否）。实测会被优化器接受成「同 bits、更低噪声」的配置反过来支配 baseline（block1 build 崩）。已排除 None。|
| 独立 fusion 驱动 `train_sequential_fusion` | 逻辑干净但要重复 PPO 循环 + 持久化 + 诊断 + 四卡接线（DRY 差）。选了在现有 `train_sequential` 里穿 ~6 个 fusion 分支。|
| 偏序用「动作提议 SF」求和 | fusion 会改变「哪些噪声点存在」，只有按 post-override 实际安装点求方差才是真实最小噪声。保留为 `NoiseOrder` 备选。|

## Consequences

**Positive**：
- 动作空间大幅缩小；探索退化成 option 上的普通 categorical 熵，对小离散空间正好合适。
- 复用现有 PPO/持久化/诊断/四卡脚手架；single-shot / 现有路径 bit-for-bit 不变（flag 默认 off）。
- map 只含 valid 配置 → SF 侧 invalid 剪枝基本不再需要。
- 服务器 F1 smoke 验证：300 episode、fusion 模式启用、四卡 probe ~3.99x、invalid_steps=0、无坍塌、零 fusion-path 错误。

**Negative / trade-offs**：
- map 是离线产物，skeleton 再生成后要重建（~分钟级、入 git）。
- 「min-noise per fusion」把 SF→bits 那条成本杠杆折进了「max-SF」一端：不能 fuse 的 block
  （block1/block4 实测 fusion 恒 0）只剩 1 个 option（=baseline），RL 对它们只能调 K。
- 偏序 `NoiseOrder` 是「暂定」（方差和），后续可能调整。

**Things to watch / future re-evaluation triggers**：
- 若想给 block1/block4 也加成本杠杆（现在只能调 K），需要调整偏序 / option 选取（暴露 SF-reduction option）。
- 若某 block-type 的 option 数病态大或只有 1 个 fusion_count，重审 builder。
- skeleton 再生成 → 重建 map（builder 的 baseline=option0 断言会守住正确性）。

## References

- Linked code: `blb_stage2_rl/fusion_count_map.py`, `blb_stage2_rl/fusion_enum.py`,
  `scripts/blb_build_fusion_count_map.py`, `blb_stage2_rl/action_space.py`
  (`fusion_step_schedule` / `expand_fusion_step_action`), `blb_stage2_rl/sequential_env.py`,
  `blb_stage2_rl/sequential_runner.py` (fusion branch).
- Spec: `docs/superpowers/specs/2026-06-03-stage2-fusion-count-action-design.md`
- Plan: `docs/superpowers/plans/2026-06-03-stage2-fusion-count-action.md`
- Related: ADR-001 (per-block sequential PPO), ADR-002/007 (reward), ADR-006 (F0/F1/F4 ladder).
