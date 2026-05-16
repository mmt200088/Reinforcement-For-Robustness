# ADR-006: F0 / F1 / F4 三级 fidelity ladder（删除 F2 / F3）

- **Status**: Accepted
- **Date**: 2026-05-16
- **Supersedes**: 早期文档里的 "F0-F4 five-tier ladder" 表述
- **Tags**: evaluation, fidelity, candidate-store

## Context

最早设计 fidelity ladder 是希望"候选 schedule 一步步升级":
- F0: optimizer-only（no model forward）
- F1: small probe
- F2: medium probe
- F3: confirmation (large probe + multi-seed)
- F4: full validation_full final eval

实践中观察到：
- F2 / F3 没人真正用 — 它们没有独立的命令行接口、没有独立的输出格式
- 训练过程产生的 reward 信号 = F1（probe 256 × MC 5）
- 真正的 "best" 选择 = F4（final eval validation_full）
- 中间没有自然的 cut-off，F2 / F3 是死代码

`candidate_store.py` 仍写 `"F0", "F1", "F2", "F3", "F4"`，导致：
- 新人读代码时以为有 5 级流程
- 文档里"F0/F1/F2 verification"实际只有 F0 / F1
- `fidelity_rank` 函数返回的 ordering 对 F2 / F3 没有调用方使用

## Decision

把 ladder 简化为 **F0 / F1 / F4 三级**：

- **F0** — optimizer-only：`Rescale_optimizer` 验 cfg `valid /
  total_bits / fusion_count`。无模型 forward。用 `scripts/
  blb_f0_scan_feasible_domain.py` 跑。
- **F1** — training-time probe：`probe_size=256` × `k_trials=5` MC 噪声
  trial。**训练循环里实时产生**，作 reward 信号。
- **F4** — final eval：`validation_full` × `final_eval_repeat_n` 次 MC，
  装真 BLB。**论文报数只能用 F4**。

代码层面：
```python
# blb_stage2_rl/candidate_store.py
FIDELITY_ORDER = {"F0": 0, "F1": 1, "F4": 2}
```

老的 JSONL 候选记录里如果有 `fidelity="F2"` / `"F3"`，`fidelity_rank`
返回 -1，candidate_store 会把它们当 legacy 处理（不可升级、不参与 rank）。

## Alternatives considered

| Option | Why rejected |
|--------|--------------|
| 保留 5 级，给 F2 / F3 加命令行接口 | 没人需要；增加测试矩阵和文档负担 |
| 完全废除 ladder，只用 F0 / F4 | F1（训练 reward）是事实存在的 fidelity 等级，不能忽略 |
| 把 F2 / F3 合并成 "F1.5 confirmation" | 没有用例需要；如果未来需要再加 |

## Consequences

**Positive**：
- 文档一致：CLAUDE.md / docs/ / HTML 都简化成 3 级
- candidate_store 的 ordering 更清晰
- 新人 onboarding 时心智模型小一档

**Negative / trade-offs**：
- 老的 candidate_store.jsonl 里的 F2 / F3 记录变成 legacy；如果你回头想
  从那些记录里挑 best，要手动改 fidelity 字段
- 失去"中间 confirmation 步骤"的语义：训练 best (F1) → 直接 F4，没有
  中间过滤。如果发现 F1 winners 在 F4 大量 collapse → 可能要加一个
  F1.5 / F2，到时候再开新 ADR

**Things to watch**：
- 如果 F1 → F4 之间一致性变差（即"训练时好的 schedule final-eval 拉跨"），
  考虑加一个中间 confirmation step

## References

- Code: `blb_stage2_rl/candidate_store.py::FIDELITY_ORDER`
- Code: `scripts/blb_make_run_manifest.py::fidelity_policy`
- CLAUDE.md "Verification: F0 → F1 → F4 fidelity ladder"
- 删除的 commit: 本 ADR 同时落地
