# Architecture Decision Records (ADRs)

每个 ADR 记录一个**重大决策**：背景（context）、决定（decision）、
后果（consequences）。

ADR 不是 design doc，也不是用户教程；它是 **"我们为什么做这个选择 / 拒绝那个选择"**
的依据。半年之后回头改时，先翻 ADR 看决策的前提是否还成立。

## 何时该写一个新 ADR？

满足任一条件就写：
1. **架构层面的决定**（模块拆分 / 数据流走向 / 状态机形状）
2. **不可逆 / 难回滚的决定**（schema 版本 / 持久化目录布局）
3. **跨多文件影响**的决定（reward 设计 / 动作空间形状）
4. **有备选方案被显式拒绝**（"也想过 A，但选了 B 因为 X"）
5. **基于实验 / 数据决定**（要把"为什么这个数字"留下来）

bug fix、单文件 refactor、小工具脚本不需要 ADR；commit message + code
comment 就够了。

## 编号约定

- 文件名：`ADR-NNN-<short-kebab-slug>.md`，NNN 从 001 起递增
- 一旦合并就 **不改编号**；废弃用 superseded 链接

## 状态机

```
Proposed → Accepted → (Superseded by ADR-XXX | Deprecated)
                    ↘ Rejected
```

- **Proposed**：草稿，欢迎反对
- **Accepted**：已生效；改它要先写新 ADR
- **Superseded**：被新 ADR 替代；保留作历史
- **Deprecated**：废弃但未被替代（很少见）
- **Rejected**：评估过、否决了；保留是为了不让后人再次提出

## 模板

见 `_TEMPLATE.md`。

## 当前 ADR 索引

| #   | 标题                                              | 状态      | 日期       |
|-----|---------------------------------------------------|-----------|------------|
| 001 | Per-block sequential PPO（vs single-shot 577-dim） | Accepted              | 2026-05-15 |
| 002 | Hard-priority reward（vs weighted sum）           | Superseded by ADR-007 | 2026-05-12 |
| 003 | Per-block 差异化 K baseline {13,10,13,10,13}      | Accepted              | 2026-05-15 |
| 004 | static_skeletons 作为唯一 BLB Stage-2 baseline 源 | Accepted              | 2026-05-14 |
| 005 | SF/K-first 人类可读输出 schema                    | Accepted              | 2026-05-16 |
| 006 | F0 / F1 / F4 三级 fidelity ladder（去掉 F2/F3）   | Accepted              | 2026-05-16 |
| 007 | v2-style clipped-shaping + tier-bonus reward      | Accepted              | 2026-05-18 |
| 008 | Per-block fusion-count action（vs 逐槽 SF）        | Accepted              | 2026-06-03 |
| 009 | Stage-2 确定性播种 + episode 并行（均匀档位部分当日撤回，保持 hybrid） | Accepted（D3 撤回） | 2026-06-10 |
| 010 | step-1×15 SF 档位 + 直连 replan 快枚举（金vs快等价门禁） | Accepted | 2026-06-11 |
| 011 | Fusion 觅取奖励（P3 预算拆分）+ 周期性强制 fusion 探针 | Accepted（探针设计被 012 取代） | 2026-06-11 |
| 012 | 可导航精度边界（近界渐变+边缘复测）+ ε 探索下限 + policy-K 探针 | Accepted（near-miss tier 被 013 取代） | 2026-06-12 |
| 013 | Stage-1 式 log-barrier 精度边界（取代 near-miss tier + 线性 P3 margin） | Accepted（barrier 仍用，被 014 补强） | 2026-06-13 |
| 014 | 结构性反失控 fusion 成本（饱和）+ 崩溃调试落盘（被 4th-60k 热崩溃触发） | Accepted（饱和被 015 退役；调试落盘保留） | 2026-06-14 |
| 015 | 连续有界 reward（移植 Stage-1）+ std 倍率稳定性门（baseline_std×tol，移植原始 Stage-2，跑宽松 5×）+ Stage-1 cosine 探索 + 严格可行性选择 | Accepted | 2026-06-14（稳定性门 06-15 更正为倍率） |
