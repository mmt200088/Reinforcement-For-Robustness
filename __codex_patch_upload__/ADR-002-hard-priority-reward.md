# ADR-002: Hard-priority reward（拒绝 weighted-sum）

- **Status**: Accepted
- **Date**: 2026-05-12
- **Tags**: rl-design, reward, evaluation

## Context

部署侧噪声配置要同时满足三个约束：
1. **accuracy**：BLB 模型 forward 后的 accuracy 不能比 baseline 低超过
   tolerance（MRPC ±0.5pp）
2. **stability**：MC 噪声 trial 之间的 loss_std 不能超过阈值
3. **cost**：通过 1-2 后，剩下的 schedule 之间按 modulus chain 开销
   （total_bits + fusion_count + avg_truncation_k）排名
4. **optimizer feasibility diagnostic**：Rescale_optimizer 报告的
   `invalid_chain` 不能跳过模型 forward，只能在 cost 层作为 feasibility
   penalty / 诊断项参与。

Naive 设计：把这些做 weighted sum 当 reward：
```
reward = -acc_penalty - stab_penalty - cost_penalty
```

观察到的问题：
- PPO 会找到**"违反 accuracy 但 cost 极低"**的 cheating schedule
- 调权重让 cost 不能"补偿"accuracy → 权重必须设得相对悬殊（差 10⁴+）
- 数值不稳定：reward 范围爆炸到 [-1e6, 0]，value head 学不动

## Decision

把 reward 做成 **hard priority**：

```
if acc < acc_threshold:     reward = -P1_PENALTY - acc_violation_scaled
elif loss_std > stab_thr:   reward = -P2_PENALTY - stab_violation_scaled # 量级 -1
else:                       reward = w_bits·bits_drop + w_fusion·fusion_drop
                                   + w_k·k_drop
                                   - optimizer_invalid_penalty(if any)
```

**关键性质**：
- 低优先级（cost）的最大 reward 永远 **小于** 高优先级（accuracy）的最小
  penalty。即 cost 的优秀永远 **不能补偿** accuracy 的违反
- Rescale_optimizer 是 cost/feasibility 诊断源；`any_invalid=True` 不允许
  跳过 BLB 安装和模型 forward，也不允许排在 accuracy/stability 前面
- 同时 **不放弃** lower-tier 信号：每个 tier 内部仍有连续 scaled
  violation 项

best 候选选择不直接用 PPO 总 reward 排，而是用元组 rank key：
```python
(acc_violation, stab_violation, optimizer_invalid_flag, normalized_cost, ...)
```
（见 `blb_stage2_rl/candidate_store.py::candidate_rank_key`）。

## Alternatives considered

| Option | Why rejected |
|--------|--------------|
| Weighted-sum + 大权重差 | 数值不稳定 + 仍存在边界 cheating（见 Context） |
| Lagrangian / Penalty method | 工程复杂度高；不需要 dual variable 的精细控制 |
| Constrained policy optimization (CPO) | PPO 实现成熟；CPO 还要 trust-region 估计 |
| Hierarchical RL（先学 feasibility，再学 cost） | 训练时间 2x，先不上 |

## Consequences

**Positive**：
- best 选择无歧义：tuple 元素逐层比较
- 数值稳定：reward 范围由三层 tier 控制，cost 不越级
- 加新约束只要在 tuple 前面插一项

**Negative / trade-offs**：
- 三个 tolerance（acc/stab）是 hyperparameter，对最终选择敏感
- 没有 cost vs accuracy 的 Pareto front；只有 "feasibility 通过后的 cost
  最小"

**Things to watch**：
- 如果三个 tolerance 调得太严 → 可能 PPO 永远找不到 valid 解；放宽
- 如果发现 PPO 长期卡在 priority1（accuracy）层 → 检查 `acc_threshold` 是
  否设得不切实际，或 stage1 GELU/Softmax degree 是否过低（导致 baseline
  accuracy 本身就掉很多）

## References

- Code: `blb_stage2_rl/reward.py`, `candidate_store.py::candidate_rank_key`
- AGENTS.md "Critical mental model" reward item
- 旧 weighted-sum 尝试见 git log 中 2026-04 月份的 commits（早期版本）
