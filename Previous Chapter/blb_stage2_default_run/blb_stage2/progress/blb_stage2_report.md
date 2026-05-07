# BLB Stage 2 RL 训练报告（最终版）

- 运行名（run_basename）: `.`
- Profile（数据集）: `mrpc`
- 生成时间: 2026-05-07T01:39:50.197993
- 训练时长: 2.4 秒（约 0.0 分钟）
- Episode 进度: 5 / 5
- 模数链 invoker: `in_process_real`

## 1. Reward 概览

- 最优 reward (best): **-198.000000**
- 全程 episode reward 均值: -198.0000
- 全程 episode reward 最大值: -198.0000
- 全程 episode reward 最小值: -198.0000
- 全程 episode reward 标准差: 0.0000

## 2. 最优 reward 拆解

| 字段 | 值 |
|------|------|
| `reward` | -198.0 |
| `priority` | 1 |
| `invalid` | False |
| `r_bits` | 0.0 |
| `r_fusion` | 0.0 |
| `r_k` | 0.0 |
| `bits_drop` | 0.0 |
| `k_drop` | 0.0 |
| `fusion_count` | 0.0 |
| `acc_violation` | 0.49 |
| `stab_violation` | 0.0 |

## 3. Baseline（全 max action）对照

| 字段 | 值 |
|------|------|
| `total_bits_sum` | 968 |
| `total_fusion_count` | 0 |
| `avg_k` | 13.0 |
| `loss_mean` | 0.692154049873352 |
| `metric1_mean` | 0.5 |

## 4. Reward 权重

| 字段 | 值 |
|------|------|
| `w_bits` | 0.03333333333333333 |
| `w_fusion` | 1.0 |
| `w_k` | 1.0 |
| `acc_threshold` | 0.49 |
| `stab_threshold` | 0.001 |

## 5. 最优 action 向量

- 长度: 147

```
0, 0, 0, 0, 1, 2, 3, 2, 1, 4, 3, 1, 3, 2, 0, 1, 0, 1, 1, 2, 2, 1, 0, 2, 3, 0, 1, 0, 2, 1, 0, 2, 2, 0, 0, 2, 1, 1, 3, 1, 4, 2, 2, 1, 0, 2, 1, 3, 0, 2, 0, 0, 2, 1, 0, 0, 2, 4, 0, 2, 3, 2, 0, 3, 2, 0, 0, 1, 3, 2, 1, 2, 4, 4, 2, 0, 0, 2, 3, 3, 2, 3, 3, 2, 1, 2, 0, 0, 2, 0, 2, 1, 0, 2, 0, 2, 3, 2, 1, 2, 2, 1, 2, 0, 4, 0, 2, 2, 1, 1, 3, 0, 5, 2, 3, 0, 1, 0, 2, 0, 1, 0, 0, 0, 0, 2, 1, 3, 3, 5, 1, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 2, 2, 1
```

---

> 持久化目录：`Parting Chapter/<run>/blb_stage2/progress/`。live checkpoint / final checkpoint / best_cfg.pkl / 状态板 / 训练曲线（PNG + NPZ）/ 本报告 都在该目录下。