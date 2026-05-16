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
| 001 | Per-block sequential PPO（vs single-shot 577-dim） | Accepted  | 2026-05-15 |
| 002 | Hard-priority reward（vs weighted sum）           | Accepted  | 2026-05-12 |
| 003 | Per-block 差异化 K baseline {13,10,13,10,13}      | Accepted  | 2026-05-15 |
| 004 | static_skeletons 作为唯一 BLB Stage-2 baseline 源 | Accepted  | 2026-05-14 |
| 005 | SF/K-first 人类可读输出 schema                    | Accepted  | 2026-05-16 |
| 006 | F0 / F1 / F4 三级 fidelity ladder（去掉 F2/F3）   | Accepted  | 2026-05-16 |
