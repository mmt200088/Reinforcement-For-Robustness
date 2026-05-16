# ADR-003: Per-block 差异化 K baseline {B1=13, B2=10, B3=13, B4=10, B5=13}

- **Status**: Accepted
- **Date**: 2026-05-15
- **Tags**: baseline, action-space, k-truncation

## Context

每个 BLB block 末尾有一个截断槽 (kind `K`)，决定 block 输出在 MPC↔HE
往返时**截断多少 bits**。K 越大 → 截断越激进 → 通讯量越少 → 但 truncation
噪声越大。

合法取值 `K_LEVELS = (8, 9, 10, 11, 12, 13)`。原来 baseline 全取 K=13（最
激进），让 RL 决定要不要保守。

观察：
- Block 2（LN-tail / QKV / QK BSGS）和 Block 4（Softmax·V / Wo /
  post-attn LN）对 attention 输出精度敏感
- Block 1（FFN2 / LN-head）、Block 3（Softmax exp 迭代平方）、Block 5
  （LN-tail / Wffn1 / GELU）对截断噪声相对鲁棒
- 全 K=13 baseline 让 RL 经常需要从"几乎不可行"出发，invalid 率早期
  > 50%

如果 baseline 自己就考虑了 attention 敏感性、把 B2 / B4 设保守一点，
RL warmstart 会更稳，per-step cost shaping 也能更早 carry 信号。

## Decision

新的 per-block K baseline：

| Block | Baseline K | 含义 |
|-------|-----------:|------|
| 1 (FFN2 / LN-head) | **13** | 浅 block 稳健，激进截断省通讯 |
| 2 (LN-tail / QKV / QK) | **10** | attention 一端，保守 |
| 3 (Softmax exp) | **13** | 迭代平方对截断不敏感 |
| 4 (Softmax·V / Wo / LN-head) | **10** | attention 另一端，保守 |
| 5 (LN-tail / Wffn1 / GELU) | **13** | GELU 多项式稳健 |

L=12 时 baseline avg_k = (4×46 + 11×59) / 59 ≈ **11.78**（之前是 13.0）。

实现：`blb_stage2_rl/action_space.py::BASELINE_K_BY_BLOCK`。
`make_all_max_action_vector` 按 block 类型解码；
`baseline_bootstrap.static_skeletons_baseline_to_action` 用相同表算
`cost_stats.avg_k`。

## Alternatives considered

| Option | Why rejected |
|--------|--------------|
| 保持 K=13 全 max | invalid 率早期太高（见 Context） |
| 让 RL 自学（baseline K=8 最保守） | warmstart 偏置太弱；cost 信号长期负偏 |
| Per-layer 差异（深层 vs 浅层）| 缺少先验数据；过拟合训练集风险 |
| 用 stage1 degree 推 K（degree 高 → K 高）| degree 已经反映多项式精度；耦合两个 hyper 难解释 |

## Consequences

**Positive**：
- baseline 本身已合理，warmstart 偏置更接近"safe 起点"
- per-step cost shaping 的 zero-point 落在 avg_k=11.78，rewards
  分布更对称
- 让 stakeholder（reviewer / 合作者）易理解："我们承认 attention 敏感"

**Negative / trade-offs**：
- 引入 block-level prior，丧失了一点"纯 RL search" 的纯粹性
- avg_k 变了 → 老的 reward weight calibration 公式需要重跑（已经处理）
- 数字 `{13,10,13,10,13}` 来自工程经验，没有理论推导支撑

**Things to watch**：
- 如果换模型 / 换 task → 这套数字可能不再合适；ADR 要 supersede
- 如果某个 block 在大量 run 里 RL 总是从 baseline 偏开 → 提示 baseline
  设错了，应该按那个方向调

## References

- Code: `blb_stage2_rl/action_space.py::BASELINE_K_BY_BLOCK`
- Code: `blb_stage2_rl/baseline_bootstrap.py::static_skeletons_baseline_to_action`
- HTML 文档第 3.2 节: `reports/session_summary/blb_stage2_rl_guide.html`
