# BLB Stage-2 RL 动作槽位 ↔ Rescale_optimizer 节点对应表

本文档把 RL 一侧每个动作槽位（slot）映射到 `Rescale_optimizer` 一侧的图节点
（node）/ skeleton stage / rotation 候选点。用作两边对接和回归校验的权威单一参考。

读者：BLB RL 这边做接口的同事 + Rescale_optimizer 这边做图与重规划的同事。

字段约定：

- **`field`**：BLB cfg 的 Python 字段名（来自 `function_handler.Block{N}NoiseConfig`）
- **`slot_label`**：紧凑日志标签 `L{layer}.B{block}.{kind}.{short}`（见
  `blb_stage2_rl/action_space.make_slot_label`）
- **`kind`**：`F` fresh / `W` weight encode / `M` mask encode / `S` scalar encode /
  `R` rescale / `K` block-end MPC truncation
- **`RO node`**：`Rescale_optimizer` 图节点名（按 `block{N}_<key>.json` 给定）。
  `—` 表示该槽位**只在明文模型噪声仿真路径**，不进入 `delta_overrides`，对模数链
  开销无影响（但会影响精度/稳定性 reward）。
- **`skeleton i`**：在该 graph 的 `skeleton` 上对应的 stage 索引；只有 fresh + 关键
  rescale 上 skeleton。`—` 表示不上 skeleton（rescale 但没被选为 cut point）。
- 表格里 `L*` 表示"对所有 layer 通用"，`L0`/`L11` 是具体层。

---

## 0. 前置（block 之外）

| RL field | slot_label | kind | RO node | skeleton i | 说明 |
|---|---|---|---|---|---|
| `first_input_sf` | `L0.first_input.F` | F | `—` | `—` | **DEPRECATED (2026-05)**。原本用于 layer-0 input from embedding 的 fresh 噪声；语义更新后认为"第一个 HE 配置无损"，不再注入。动作向量保留此槽位仅为 backward compat，`describe_action_vector` 标 `effective=False`；`BLBNoiseRLBridge.apply()` 完全忽略它。 |

---

## 1. Block 1 — post-FFN GELU output → LayerNorm var

**Graph 文件**：`block1_<profile>.json`（每个 dataset 一份图，与 GELU/Softmax degree 无关）。
**RL 一侧 N 默认**：`8192`。**fixed-add**：1 fresh + 3 encode；**RL-selectable**：4 rescale + 1 K + 4 rotation flag。

> **⚠ layer-0 不安装**：layer 0 没有上游 FFN2（X 直接来自 embedding 进 block 2 的
> LayerNorm），所以**整个 block 1 噪声在 layer 0 整体不安装**，对应的
> `block1_<dataset>_L0` 也不发给 Rescale_optimizer。下方表里 `L*` 应理解为
> `L1..L11`；layer 0 的 9 个 block 1 槽位在动作向量里保留但 `effective=False`。

| RL field | slot_label | kind | RO node | skeleton i | 说明 |
|---|---|---|---|---|---|
| `gelu_out_fresh` | `L*.B1.F.gelu_out` | F | source / `ctct_gelu_out`（图入口） | 0 | GELU 输出 fresh 噪声。 |
| `wffn2_encode` | `L*.B1.W.wffn2` | W | `ctpt_ffn2` (CTPT_MUL) | — | W_ffn2 plaintext encode。 |
| `mean_inv_d_encode` | `L*.B1.S.mean_invd` | S | `ctpt_inv_d_1` (CTPT_MUL) | — | μ 步骤的标量 1/D encode。 |
| `var_inv_d_encode` | `L*.B1.S.var_invd` | S | `ctpt_inv_d_2` (CTPT_MUL) | — | var 步骤的标量 1/D encode。 |
| `wffn2_result_rescale` | `L*.B1.R.wffn2_r` | R | `ctct_ffn2_rescale`* | — | W_ffn2·X 后的 rescale；可选。 |
| `mean_result_rescale` | `L*.B1.R.mean_r` | R | `ctct_mean_rescale`* | 2 | μ 计算后的 rescale；on skeleton。 |
| `square_result_rescale` | `L*.B1.R.square_r` | R | `ctct_ext_square` (CTCT_MUL, delta=`"x2"`) | — | (X−μ)² 之后；delta 固定为乘数 2。 |
| `var_result_rescale` | `L*.B1.R.var_r` | R | `ctct_var_rescale`* | 4 | var 计算后的 rescale；on skeleton。 |
| `output_truncation_k` | `L*.B1.K` | K | `—` | `—` | block 末 MPC↔HE 截断 k。 |

**Rotation 候选点**（共 4 个；scaling factor 取自前面紧邻的 fresh / rescale）：

| flag (cfg) | 紧邻在 | 受控源 |
|---|---|---|
| `rotation_after_gelu_out_fresh` | gelu_out fresh 之后 | `gelu_out_fresh.scaling_factor` |
| `rotation_after_wffn2_result_rescale` | W_ffn2·X 的 rescale 之后 | `wffn2_result_rescale.scaling_factor` |
| `rotation_after_mean_result_rescale` | μ 的 rescale 之后（紧跟前一个 rotation） | `mean_result_rescale.scaling_factor` |
| `rotation_after_var_result_rescale` | var 的 rescale 之后 | `var_result_rescale.scaling_factor` |

`*` 标记：实际 graph 里这个 rescale 节点可能合并到上游运算节点的 `out_rescale_target`，不一定有独立节点名。RL 端通过 `t_new` skeleton 对齐。

---

## 2. Block 2 — LayerNorm tail → QK^T

**Graph 文件**：`block2_<profile>.json`。**RL 一侧 N 默认**：`16384`。
**Wq/Wk 共享 SF**：BLB 协议约束。RL cfg 仍保留 `wq_encode`/`wk_encode` 两个字段，但
`default_block2_cfg_to_delta` 只取 `wq_encode.scaling_factor` 写到 `ctpt_wq_wk` 节点
（共享）。Wv 在 graph 里**没有节点**（Wv 路径不进模数链 cost；只进明文噪声）。

| RL field | slot_label | kind | RO node | skeleton i | 说明 |
|---|---|---|---|---|---|
| `inv_std_fresh` | `L*.B2.F.inv_std` | F | source（inv_std 入口） | 0 | LayerNorm 1/std fresh。 |
| `x_centered_fresh` | `L*.B2.F.x_centered` | F | source（x−μ 入口） | — | (X−μ) fresh。 |
| `gamma_encode` | `L*.B2.M.gamma` | M | `ctpt_gama1` (CTPT_MUL) | — | γ encode。 |
| `wq_encode` | `L*.B2.W.wq` | W | `ctpt_wq_wk` (CTPT_MUL, **shared**) | — | Wq encode（与 wk 共享 SF）。 |
| `wk_encode` | `L*.B2.W.wk` | W | `ctpt_wq_wk` (CTPT_MUL, **shared**) | — | Wk encode（与 wq 共享 SF）。 |
| `wv_encode` | `L*.B2.W.wv` | W | `—` | `—` | Wv 不进 RO 图；仅明文噪声。 |
| `kt_mask1_encode` | `L*.B2.M.kt_mask1` | M | `ctpt_rotKT_mask1` (CTPT_MUL) | — | K^T BSGS mask 1 encode。 |
| `kt_mask2_encode` | `L*.B2.M.kt_mask2` | M | `ctpt_rotKT_mask2` (CTPT_MUL) | 5 | K^T BSGS mask 2 encode；**与 stage 5 关联**。 |
| `q_mask1_encode` | `L*.B2.M.q_mask1` | M | `—` | `—` | 仅明文噪声。 |
| `q_mask2_encode` | `L*.B2.M.q_mask2` | M | `—` | `—` | 仅明文噪声。 |
| `qkt_merge_mask_encode` | `L*.B2.M.qkt_merge_mask` | M | `ctpt_mask` (CTPT_MUL) | 7 | Q*K^T 合并 mask；**stage 7**。 |
| `normalize_result_rescale` | `L*.B2.R.normalize_r` | R | `ctct_x_mean_over_std` (CTCT_MUL, `"x2"`) | — | (X−μ)·(1/std) 之后；delta=`"x2"`。 |
| `gamma_result_rescale` | `L*.B2.R.gamma_r` | R | `ctct_gamma_rescale`* | 2 | γ 之后 rescale；on skeleton。 |
| `wk_result_rescale` | `L*.B2.R.wk_r` | R | `ctct_wk_rescale`* | — | Wk·X 之后。 |
| `wq_result_rescale` | `L*.B2.R.wq_r` | R | `ctct_wq_rescale`* | — | Wq·X 之后。 |
| `wv_result_rescale` | `L*.B2.R.wv_r` | R | `—` | `—` | Wv 路径不进 RO。 |
| `kt_mask1_result_rescale` | `L*.B2.R.kt_mask1_r` | R | `ctct_kt_mask1_rescale`* | — | K^T·mask1 之后。 |
| `kt_mask2_result_rescale` | `L*.B2.R.kt_mask2_r` | R | `ctct_kt_mask2_rescale`* | — | K^T·mask2 之后。 |
| `q_mask1_result_rescale` | `L*.B2.R.q_mask1_r` | R | `—` | `—` | 仅明文路径。 |
| `q_mask2_result_rescale` | `L*.B2.R.q_mask2_r` | R | `—` | `—` | 仅明文路径。 |
| `qkt_matmul_rescale` | `L*.B2.R.qkt_matmul_r` | R | `ctct_preprocess_qkt` (CTCT_MUL, `"x2"`) | — | Q·K^T matmul；delta=`"x2"`。 |
| `qkt_merge_mask_result_rescale` | `L*.B2.R.qkt_merge_mask_r` | R | `ctct_qkt_merge_mask_rescale`* | — | 合并 mask 之后。 |
| `output_truncation_k` | `L*.B2.K` | K | `—` | `—` | block 末截断。 |

**Rotation 候选点**（5 个）：

| flag | 紧邻在 | 受控源 |
|---|---|---|
| `rotation_after_normalize_result_rescale` | (X−μ)·(1/std) 的 rescale 之后 | `normalize_result_rescale.scaling_factor` |
| `rotation_after_wq_wk_wv_rescale` | Wq/Wk/Wv 三个 rescale 之后（三分支共一旗标） | 上游 rescale SF（取 wq） |
| `rotation_after_qk_mask1_rescale` | 第一个 mask × Q/K^T 之后 | `kt_mask1_result_rescale` / `q_mask1_result_rescale` |
| `rotation_after_qk_mask2_rescale` | 第二个 mask × Q/K^T 之后 | `kt_mask2_result_rescale` / `q_mask2_result_rescale` |
| `rotation_after_qkt_matmul_rescale` | Q·K^T matmul 的 rescale 之后 | `qkt_matmul_rescale.scaling_factor` |

---

## 3. Block 3 — Softmax 指数近似

**Graph 文件**：`block3_exp_n<degree>.json`，degree ∈ {2,3,4,5,6}（与每层 softmax 阶数对齐）。
**RL 一侧 N**：degree=2 → `8192`；degree ≥ 3 → `16384`。
**RL-selectable**：1 fresh + 1 encode + (degree+1) rescale + 1 K，**rotation = 0**（block 3 无 rotation）。

| RL field | slot_label | kind | RO node | skeleton i | 说明 |
|---|---|---|---|---|---|
| `x_fresh` | `L*.B3.F.x` | F | source（softmax 输入入口） | 0 | x fresh（softmax input）。 |
| `inv_2n_encode` | `L*.B3.S.inv_2n` | S | `ctpt_inv_2n` (CTPT_MUL) | — | 1/2^n encode。 |
| `x_inv_2n_rescale` | `L*.B3.R.x_inv_2n_r` | R | `ctct_x_inv_2n_rescale`* | — | x·(1/2^n) 之后 rescale。 |
| `square_rescales[0]` | `L*.B3.R.sq0` | R | `ctct_square_1` (CTCT_MUL, `"x2"`) | 2 | 第 1 次平方；on skeleton。 |
| `square_rescales[1]` | `L*.B3.R.sq1` | R | `ctct_square_2` (CTCT_MUL, `"x2"`) | 3 | 第 2 次平方。 |
| `square_rescales[2]` | `L*.B3.R.sq2` | R | `ctct_square_3` (CTCT_MUL, `"x2"`) | 4 (deg=4) | 第 3 次平方（deg≥3 启用）。 |
| `square_rescales[3]` | `L*.B3.R.sq3` | R | `ctct_square_4` (CTCT_MUL, `"x2"`) | 5 (deg=4) | 第 4 次平方（deg=4 启用）。 |
| `output_truncation_k` | `L*.B3.K` | K | `—` | `—` | block 末截断。 |

`describe_action_vector` 会用 `_is_action_field_effective` 标记某层 `square_rescales[k]`
当 `softmax_degree <= k` 时为 `effective=False`（inactive 槽位）。`degree=5/6` 在
`DEFAULT_CFG_TO_T_NEW_MAP` 用最后一项 fallback（cfg 只有 max degree-1=4 个 square 槽）。

---

## 4. Block 4 — Softmax × V → post-attention LayerNorm var

**Graph 文件**：`block4.json`（无 profile / degree 后缀，所有数据集共用）。
**RL 一侧 N 默认**：`16384`。
**RL-selectable**：2 fresh + 6 encode + 8 rescale + 1 K + 6 rotation。

| RL field | slot_label | kind | RO node | skeleton i | 说明 |
|---|---|---|---|---|---|
| `softmax_out_fresh` | `L*.B4.F.softmax_out` | F | source | 0 | softmax 输出 fresh。 |
| `v_fresh` | `L*.B4.F.v` | F | source | — | V 矩阵 fresh。 |
| `softmax_out_mask_encode` | `L*.B4.M.softmax_out_mask` | M | `ctpt_mask2` (CTPT_MUL, **shared**) | — | softmax_out × mask；与 v_mask 共享节点。 |
| `v_mask_encode` | `L*.B4.M.v_mask` | M | `ctpt_mask2` (CTPT_MUL, **shared**) | — | V × mask；共享 SF（取 softmax_out_mask）。 |
| `softmax_v_mask_encode` | `L*.B4.M.softmax_v_mask` | M | `ctpt_mask` (CTPT_MUL) | — | softmax×V 后的合并 mask。 |
| `ln_mean_inv_d_encode` | `L*.B4.S.ln_mean_invd` | S | `ctpt_inv_d_1` (CTPT_MUL) | — | post-attn LN μ 步骤 1/D。 |
| `ln_var_inv_d_encode` | `L*.B4.S.ln_var_invd` | S | `ctpt_inv_d_2` (CTPT_MUL) | — | post-attn LN var 步骤 1/D。 |
| `wo_encode` | `L*.B4.W.wo` | W | `ctpt_wo_attnout` (CTPT_MUL) | — | W_o encode。 |
| `softmax_out_mask_rescale` | `L*.B4.R.softmax_out_mask_r` | R | rescale-after-mask2* | — | softmax_out × mask 之后。 |
| `v_mask_rescale` | `L*.B4.R.v_mask_r` | R | rescale-after-mask2* | — | V × mask 之后。 |
| `softmax_v_matmul_rescale` | `L*.B4.R.softmax_v_matmul_r` | R | `ctct_rot_softmax_mul_v` (CTCT_MUL, baseline delta=39) | 2 | softmax×V matmul；MRPC 默认 delta=39。 |
| `softmax_v_mask_rescale` | `L*.B4.R.softmax_v_mask_r` | R | rescale-after-mask* | — | softmax×V × mask 之后。 |
| `wo_result_rescale` | `L*.B4.R.wo_r` | R | `ctct_wo_rescale`* | — | W_o · attnout 之后。 |
| `ln_mean_result_rescale` | `L*.B4.R.ln_mean_r` | R | `ctct_attn_mean_rescale`* | 5 | post-attn LN μ rescale；on skeleton。 |
| `ln_square_result_rescale` | `L*.B4.R.ln_square_r` | R | `ctct_square` (CTCT_MUL, `"x2"`) | — | (X−μ)² 之后。 |
| `ln_var_result_rescale` | `L*.B4.R.ln_var_r` | R | `ctct_attn_var_rescale`* | 7 | post-attn LN var；on skeleton。 |
| `output_truncation_k` | `L*.B4.K` | K | `—` | `—` | block 末截断。 |

**Rotation 候选点**（6 个）：

| flag | 紧邻在 | 受控源 |
|---|---|---|
| `rotation_after_softmax_out_mask_rescale` | softmax_out × mask 的 rescale 之后 | `softmax_out_mask_rescale.scaling_factor` |
| `rotation_after_v_mask_rescale` | V × mask 的 rescale 之后 | `v_mask_rescale.scaling_factor` |
| `rotation_after_softmax_v_matmul_rescale` | softmax × V matmul 的 rescale 之后 | `softmax_v_matmul_rescale.scaling_factor` |
| `rotation_after_softmax_v_mask_rescale` | softmax×V × mask 的 rescale 之后 | `softmax_v_mask_rescale.scaling_factor` |
| `rotation_after_wo_result_rescale` | concat·W_o 的 rescale 之后 | `wo_result_rescale.scaling_factor` |
| `rotation_after_ln_square_result_rescale` | (X−μ)² 的 rescale 之后 | `ln_square_result_rescale.scaling_factor` |

---

## 5. Block 5 — post-attention LayerNorm tail → GELU 输出

**Graph 文件**：`block5_n<gelu_degree>.json`，degree ∈ {1,2,4}。
**RL 一侧 N**：degree=1 → `8192`；degree ∈ {2,4} → `16384`。
**RL-selectable**：2 fresh + (2 GELU-shared + 1) encode + (3 + (deg-1) power + deg coeff_mul) rescale + 1 K + 2 rotation。

GELU 多项式系数 a/b/c/d/e... 共享一个 `gelu_coeff_encode.scaling_factor`（**所有项噪声分布同 σ 但
独立采样**，不能共用同一个 ε 张量）。

| RL field | slot_label | kind | RO node | skeleton i | 说明 |
|---|---|---|---|---|---|
| `inv_std_fresh` | `L*.B5.F.inv_std` | F | source | — | 1/std fresh（与 block 2 同名同语义，但 layer 位置不同）。 |
| `x_centered_fresh` | `L*.B5.F.x_centered` | F | source | 0 | (X−μ) fresh。 |
| `gamma_encode` | `L*.B5.M.gamma` | M | `ctpt_gamal` (CTPT_MUL) | — | γ encode。 |
| `wffn1_encode` | `L*.B5.W.wffn1` | W | `ctpt_wffn1` (CTPT_MUL) | — | W_ffn1 encode。 |
| `gelu_coeff_encode` | `L*.B5.M.gelu_coeff` | M | `ctpt_gelu_coeff` (CTPT_MUL) | — | GELU 多项式系数共享 encode。 |
| `normalize_result_rescale` | `L*.B5.R.normalize_r` | R | `ctct_xmean_over_std` (CTCT_MUL, `"x2"`) | 1 (deg≥2) | (X−μ)·(1/std)；on skeleton 当 deg≥2。 |
| `gamma_result_rescale` | `L*.B5.R.gamma_r` | R | `ctct_gamma_attn_rescale`* | 2 (deg=1) | γ rescale（deg=1 时 on skeleton）。 |
| `wffn1_result_rescale` | `L*.B5.R.wffn1_r` | R | `ctct_wffn1_rescale`* | 3 (deg≥2) | W_ffn1 之后。 |
| `gelu_power_rescales[0]` | `L*.B5.R.gp0` | R | `ctct_gelu_x2` (CTCT_MUL, `"x2"`) | 4 (deg=4) | x² 计算（deg≥2 启用）。 |
| `gelu_power_rescales[1]` | `L*.B5.R.gp1` | R | `—`（degree=4 时 graph 把 x³ 折进 x⁴） | — | x³（deg=4 启用，但 graph 不算独立节点）。 |
| `gelu_power_rescales[2]` | `L*.B5.R.gp2` | R | `ctct_gelu_x4` (CTCT_MUL, `"x2"`) | — | x⁴（deg=4 启用）。 |
| `gelu_coeff_mul_rescales[0..deg-1]` | `L*.B5.R.gc0..gc{d-1}` | R | 系数·x^k 之后的 rescale（**最后一项**进 skeleton） | last entry on skeleton | 多项式系数·x^k；只取最后一项进 t_new。 |
| `output_truncation_k` | `L*.B5.K` | K | `—` | `—` | block 末截断。 |

**Rotation 候选点**（2 个）：

| flag | 紧邻在 | 受控源 |
|---|---|---|
| `rotation_after_normalize_result_rescale` | (X−μ)·(1/std) 的 rescale 之后 | `normalize_result_rescale.scaling_factor` |
| `rotation_after_wffn1_result_rescale` | W_ffn1·X 的 rescale 之后 | `wffn1_result_rescale.scaling_factor` |

---

## 6. Skeleton ↔ cfg 对应（DEFAULT_CFG_TO_T_NEW_MAP 摘要）

`Rescale_optimizer.replan_with_user_actions` 接收 `t_new[r] = ` 第 r 个 stage 的目标 SF。
RL 一侧从 cfg 自动派生 `t_new`，对应表见 `rescale_optimizer_bridge.DEFAULT_CFG_TO_T_NEW_MAP`：

```
block1_<profile>:  i=0 gelu_out_fresh        i=2 mean_result_rescale     i=4 var_result_rescale
block2_<profile>:  i=0 inv_std_fresh         i=2 gamma_result_rescale    i=5 kt_mask2_result_rescale  i=7 qkt_merge_mask_result_rescale
block3_exp_n2:     i=0 x_fresh               i=2 square_rescales[0]      i=3 square_rescales[1]
block3_exp_n3:     i=0..3 (= n2 + sq[2])
block3_exp_n4:     i=0..4 (= n3 + sq[3])
block3_exp_n5/n6:  扩展，重用 sq[3] 作 fallback
block4:            i=0 softmax_out_fresh     i=2 softmax_v_matmul_rescale i=5 ln_mean_result_rescale i=7 ln_var_result_rescale
block5_n1:         i=0 x_centered_fresh      i=2 gamma_result_rescale    i=4 gelu_coeff_mul_rescales[-1]
block5_n2:         i=0 x_centered_fresh      i=1 normalize_result_rescale i=3 wffn1_result_rescale  i=5 gelu_coeff_mul_rescales[-1]
block5_n4:         i=0 x_centered_fresh      i=1 normalize_result_rescale i=3 wffn1_result_rescale i=4 gelu_power_rescales[0] i=6 gelu_coeff_mul_rescales[-1]
```

---

## 7. Rotation 绑定规则（不变量）

1. **rotation 没有独立 SF action**。RL 不对 rotation 出 categorical head；其 SF 来自前
   面紧邻的 fresh / rescale。
2. **rotation 是可选的**：cfg 上的 `rotation_after_*: bool` 决定是否插入；最终是否
   插入由 `Rescale_optimizer.new_compact_config.effective_rotations` 决定，由
   `apply_rotation_flags_to_cfg` 反写到 cfg。
3. **rotation 紧跟其前一个 fresh / rescale**：若该 rescale 没被选 / 没被执行，紧邻它
   的 rotation 也不应注入。
4. **block 3 没有 rotation**。

---

## 8. 一致性 self-test（建议）

下列断言应在 `tests/test_blb_rl_rescale_mapping.py` 里被覆盖：

1. `_BLOCK_NODE_NAME_BY_FIELD` 与 `default_block{N}_cfg_to_delta` 在每个 RL field 上
   要么同名映射，要么显式标记为"明文路径"（不在 RO 图上）。
2. `make_config_name(profile, block_idx, layer_idx, cfg)` 对每个 (block_idx, degree)
   组合返回的 `graph_key` 必须存在于 `static_skeletons_<profile>.json` 的 keys 集合中。
3. `DEFAULT_CFG_TO_T_NEW_MAP` 的每个条目在 `Block{N}NoiseConfig` 上都能 getattr 到
   一个 NoisePoint（或 tuple of NoisePoint）字段，不能是悬空字段名。
4. `apply_rotation_flags_to_cfg` 对每个 block 的 cfg，要识别且只识别 `rotation_after_*`
   这些字段。

---

## 9. 节点名 unverified 标记（`*` 后缀）

带 `*` 的 RO 节点名是**根据 cfg 字段语义+`Rescale_optimizer/configs/<profile>/*.json`
通用命名习惯**推断的；具体 graph 里这些 rescale 可能被合并到上游 `out_rescale_target`
而不再作为独立节点出现。RL 一侧只通过 `t_new[skeleton_i]` 写入 SF，不依赖这些节点
的 `delta_overrides`，因此是否独立都不影响调用结果。Rescale_optimizer 一侧若要
自己改名，只需保证 `static_skeletons_<profile>.json` 的 `cut_point_sf` 顺序与
`DEFAULT_CFG_TO_T_NEW_MAP` 一致即可。

---

## 10. 与 Rescale_optimizer 的两条调用通道

RL 与 Rescale_optimizer 之间有两条**正交**的调用通道。设计意图是：**baseline 用 JSON
慢通道做团队握手；训练时用 in-process 快通道做 per-action 模数链评估。**

| 通道 | 触发时机 | 用途 | 协议 |
|---|---|---|---|
| **JSON 文件握手** | Stage-1 完成、Stage-2 训练启动前一次 | 把 Stage-1 推荐的 (gelu/softmax) per-layer 配置发给 RO；RO 返回 BLB Stage-2 全 max baseline 配置 + 模数链合法性验证 | 见 `docs/blb_baseline_handover_protocol.md` |
| **in-process 快通道** | Stage-2 训练循环每个 episode | RL 给一个 action_vec → 解码成 cfg → 调 `replan_with_user_actions` → 返回 fusion_count / total_bits / invalid_chain | `rescale_optimizer_bridge.InProcessInvoker`（`import rescale_optimizer`，无 JSON IO） |

**重要**：训练循环禁止走 JSON 通道（哪怕 SubprocessInvoker 跑得通也不要）——
`InProcessInvoker.from_profile()` 会预加载所有 graph + baseline，每次调用是 ms 级；
JSON 子进程是几百 ms / 调用，乘以 80k episodes × 5 blocks × 12 layers 直接拖慢两个数量级。
