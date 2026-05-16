# ADR-005: SF/K-first 人类可读输出 schema

- **Status**: Accepted
- **Date**: 2026-05-16
- **Tags**: artifacts, debugging, ux

## Context

RL policy 的内部表示是 `action_vec`：长度 577 的整数列表，每个槽位
是 **action_index**（取值 0..levels-1）。但 RL 决策对应的物理量是
**scaling factor (SF)** 或 **truncation_bits (K)**。这两者通过解码规则
（`sf_from`, `K_LEVELS[idx]`）转换。

老的输出文件直接 dump action_vec：
```
最优 action 向量
长度: 577
3, 4, 5, 1, 2, 0, 5, 5, 5, 1, 1, 1, ...
```

观察到的问题：
- 看不出"layer 5 block 3 选了什么 SF"
- 调试时不知道某次失败是哪个槽位的问题
- 复现 / 手改：用户没法手动调 action_index（要查解码表）
- final-eval 输入 JSON 也是 flat int list，看不懂

## Decision

**所有用户能看到的训练产物**统一切到 **SF/K-first schema**：

```json
{
  "schema_version": "blb_v3_slots_human_v1",
  "num_layers": 12,
  "profile": "mrpc",
  "slots": [
    {
      "label": "L05.B3.K",                       // 反映模型位置
      "layer": 5, "block": 3, "kind": "K",
      "operation": "block3_output_truncation",
      "truncation_bits": 10,                      // 主要值
      "level_values": [8, 9, 10, 11, 12, 13],     // 其他合法选项
      "action_index": 2                           // 兜底 / sanity
    },
    {
      "label": "L02.B5.W.wffn1",
      "kind": "W",
      "scaling_factor": 14,
      ...
    }
  ],
  "diff_vs_baseline": [
    {"label": "L05.B3.K", "baseline_truncation_bits": 13, "best_truncation_bits": 10, "delta": -3}
  ],
  "action_vec": [3, 4, 5, ...]   // Paean 兜底兼容
}
```

**覆盖的产物**：
- `blb_stage2_status.json` `best.slots` 字段
- `blb_stage2_best_action_full.{json,md}`
- `blb_stage2_baseline_action_full.{json,md}`
- `blb_stage2_report.md` §5（"按层/block 选择" + "best vs baseline diff"）
- `diagnostics/best_action_vec.json`
- `diagnostics/top_candidates.jsonl`
- `diagnostics/diagnostics_summary.md`

**用户输入也接受**这个 schema，由 `Paean/action_grid.load_action_grid_config`
解析，支持 4 种形态：
1. 完整 slots 列表
2. slots dict（label → value）
3. base + overrides（最适合手调）
4. 老 flat action_vec（向后兼容）

转换器在 `blb_stage2_rl/action_io.py`。

## Alternatives considered

| Option | Why rejected |
|--------|--------------|
| 保持 action_index，写更好的文档说明解码规则 | 用户每次都要查表 |
| 用每槽位的 SF dict 但保留旧 action_idx 索引文件 | 双 schema 长期维护负担大 |
| 只改 markdown，JSON 保持机器形式 | 编辑器 / jq 用户读不到 SF；diff 不直观 |
| 给 action_vec 加 inline comment | JSON 不支持注释；YAML 又破坏 Paean 接口 |

## Consequences

**Positive**：
- 看 best 配置 = 翻一个 .md / 一份 .json，秒懂"哪层 block 选了什么 SF"
- 手动改配置：从 baseline 复制，改几个槽位，喂给 Paean → 直接 run
- diff_vs_baseline 字段把"RL 学到的 delta"提取出来，是最直接的研究信号
- schema_version 字段为未来升级留口

**Negative / trade-offs**：
- 每次落盘要跑一次 `describe_action_vector` + slot-list 转换，CPU 开销
  增加（但相比 model forward 可忽略）
- 旧的产物文件（accumulated 在 persistent/）schema 不一致，后处理脚本要
  兼容
- Paean 解析的 4 种形态增加测试矩阵

**Things to watch**：
- 如果 action_space.py 改 slot 命名 → 老 best_action_vec.json 的 label 失效
  → 解析报错。要在 schema_version 上 bump。
- 如果某天加 hierarchical RL → label 命名规则可能要扩展 (`L05.B3.subA.K`)

## References

- Code: `blb_stage2_rl/action_io.py`（双向转换）
- Code: `blb_stage2_rl/diagnostics.py`（slots view writer）
- Code: `Paean/action_grid.py::load_action_grid_config`（多 schema 解析）
- HTML §8.1-8.3: `reports/session_summary/blb_stage2_rl_guide.html`
- Commit `6876c13` (Add diagnostics dashboard + human-readable SF/K action JSON)
- Commit `a7d96d1` (Unify all BLB Stage-2 training artifacts to SF/K-first)
