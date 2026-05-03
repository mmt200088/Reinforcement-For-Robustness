# 噪声注入候选清单 —— 按 BLB Figure 10 的 5-block 划分梳理

> 真相来源：
> 1. **BLB 论文 Figure 10**（USENIX Security 2025；arXiv [2508.19525](https://arxiv.org/abs/2508.19525)）
> 2. **本项目用户手绘的 Block1–Block5 计算图**（已对齐 BLB Figure 10 颜色，
>    并标注每个 block 用哪一张噪声方差表 N=8192 / N=16384，以及 Q/K、gelu4
>    两个 ct*ct 必须共享 scaling factor 等协议约束）
> 3. **代码真相**：
>    - [`function_handler.py`](../function_handler.py)（`PolynomialGELU` /
>      `BertSelfAttentionWithAproximation.approximation_softmax` /
>      `NOISE_VARIANCE_TABLE_BY_N` / `get_input_noise_variance_by_N`）
>    - [`noise_targets_registry.py`](../noise_targets_registry.py)（44 个 fine-grained id）
>
> 你后面要说 *"在哪些 id 上加噪声、怎么加"* 时，直接报本文里的 `id` 即可。

---

## 0. 全局约定

### 0.1 BLB 算子分类（论文 Table 2）

| 类别 | 维度变换 | 代表算子 |
|---|---|---|
| **Identity** | (L,D) → (L,D) | `ewadd`、`ewmul` |
| **Expansion** | (L,1) → (L,D) | `sadd`、`smul` |
| **Reduction** | (L,D) → (L,1) | `sum` |
| **Transformation** | (L,D) → (L,D′) | `matmul`（`matmulcc`、`matmulcp`） |
| 非线性 | — | `cmp`、`mux`、`rec`、`rsqrt` |

记号：`ct*ct` = `matmulcc / ewmulcc`（密-密乘）；`ct*pt` = `matmulcp / ewmulcp`（密-明乘）。

### 0.2 BERT 架构

**BLB / 本项目都用 post-LN**：
"Attention → +residual → LayerNorm → FFN → +residual → LayerNorm"。

### 0.3 BLB Figure 10 的 5-block 切分

BLB 把一个 Transformer layer 的所有线性算子，跨 LayerNorm / Softmax / GeLU
等非线性边界，**重新切成 5 个可融合的 block**。Block 之间的边界是非线性算子
（`rsqrt` / `cmp` / `mux` / `rec`），它们走 MPC，不属于任何一个 block。

| Block | Figure 10 颜色 | 内容（按用户手绘图） | 用哪张噪声表 |
|---|---|---|---|
| **Block 1** | 绿 | **Wffn2** + post-FFN LN 头部（mean、(x-mean)²·1/D = variance） | `N=8192` |
| **Block 2** | 蓝 | post-FFN LN 尾部（×1/std、×γ）+ **Wq/Wk(/Wv)** + **Q·K^T** | `N=16384` |
| **Block 3** | 粉 | Softmax exp 近似主体（degree 决定步数） | softmax2 → `N=8192`；softmax3/4/5/6 → `N=16384` |
| **Block 4** | 紫 | **softmax · V**（CTCT_MUL 双起点）+ **Wo** + post-attn LN 头部 | `N=16384` |
| **Block 5** | 青 | post-attn LN 尾部（×1/std、×γ）+ **Wffn1** + GeLU 多项式（degree 决定步数） | gelu1 → `N=8192`；gelu2/4 → `N=16384` |

> 时间序：Block 5 → (rsqrt) → Block 1 → (rsqrt) → Block 2 →
> (cmp/mux/rec, Block 3 内非线性) → Block 4 → (rsqrt) → Block 5（下一层）。

### 0.4 噪声方差表（多 N）—— `function_handler.py`

代码里现在并存两套表：

| 表 | 来源 | 用途 |
|---|---|---|
| `INPUT_NOISE_VARIANCE_TABLE` (legacy) | 旧实验沿用的 σ² | 现有 `replace_layer_*_noise()` 系列继续走这条；保持向后兼容 |
| `NOISE_VARIANCE_TABLE_BY_N` (新) | `noise_std_table.csv` 的 (σ_enc, σ_fresh, σ_rs) **平方** | BLB-aware 注入；按 block 选 N=8192 / 16384 |

新查表 API：
```python
from function_handler import (
    NOISE_VARIANCE_TABLE_BY_N,           # {N: {scale: {dist: σ²}}}
    NOISE_TABLE_ALLOWED_N,               # (8192, 16384)
    get_input_noise_variance_by_N,       # σ² 查表
    add_gaussian_noise_by_N,             # 直接给 tensor 加 N(0, σ²)
)

variance = get_input_noise_variance_by_N(scale, "fresh", N=16384)   # σ²
noisy    = add_gaussian_noise_by_N(x, scale, "fresh", N=16384)
```

scale_bits ∈ [10, 46]，distribution ∈ {`encoding`, `fresh`, `rescale`}。

### 0.5 用户手绘图新增的几条关键约束

1. **Block 2：Q/K 必须共享 scaling factor**
   > "在这个位置会生成 Q 和 K，Q 和 K 保持动作选择一致，加噪声一致"
   注册表里 `attn.q_proj.shared_with == ["attn.k_proj"]`（反之亦然）。

2. **Block 4：rot_softmax · V 是双起点的 CTCT_MUL，主链 / V 可独立选噪声**
   > "我们主链设置为左边的 softmax，而 *V 在 rescale skeleton 中体现为
   > other_ct_scale_bits …… V 和 mask 都是可以选动作的，rot softmax 和 V 分别添加噪声"

   配置字段示例：
   ```json
   {
       "name": "ctct_rot_softmax_mul_v",
       "type": "CTCT_MUL",
       "scale_delta_bits": 0,
       "other_ct_scale_bits": 39
   }
   ```
   注册表里 `attn.probs_v_matmul.current_noise == "softmax_probs / value_after_softmax"`，
   两个噪声键独立——已经满足这个语义。

3. **Block 5 gelu4：两个 ct*ct 必须共享 scaling factor**
   > "选相同 scaling factor，加相同噪音"
   注册表里 `ffn.gelu.power.x2.shared_with == ["ffn.gelu.power.x3x4"]`。

4. **🚧 仍 TBD（用户下次补充）**：
   - Block 2 里 `ct*ct: Q` 的精确生成步骤
   - Block 4 里 `V` 输入怎么从更早 block 流过来

---

## 1. Block 1 ＝ Wffn2 + post-FFN LN head（`N=8192`）

> 用户手绘 Block1 计算图

### 1.1 计算流（按时间顺序）

| # | BLB 算子 | 输入 1 | 输入 2 | 输出 | 项目 id |
|---|---|---|---|---|---|
| 1 | Bs Rotation | `GELU_out` (ct) | — | rotated | (rotation, 不算乘法) |
| 2 | **`ct*pt` = matmulcp** | rotated | `Wffn2 : 20` (pt) | ffn2_out | **`ffn.output_proj`** ✅ `wffn2` |
| 3 | Gs Rotation | ffn2_out | — | rotated | — |
| 4 | **Rotation Sum1: 3 次** = `sum` | rotated | — | sum | (sum, 不算乘法) |
| 5 | **`ct*pt` = smulcp** (× 1/D) | sum | `1/D : 20` (pt) | mean | **`ffn.layernorm.head.mean_smul`** |
| 6 | **`ct*pt` = ewmulcp** (× square mask) | x | `square mask` (pt) | x_centered (BLB 协议) | **`ffn.layernorm.head.center_ctpt`** |
| 7 | **`ct*ct` = ewmulcc** | x_centered | x_centered | (x − μ)² | **`ffn.layernorm.head.square_ctct`** |
| 8 | **Rotation Sum2: 3 次** = `sum` | (x−μ)² | — | sum | — |
| 9 | **`ct*pt` = smulcp** (× 1/D) | sum | `1/D : 20` (pt) | **(x−μ)²·1/D = variance** | **`ffn.layernorm.head.var_smul`** |
| → | (rsqrt 走 MPC，输出 1/std 进入 Block 2) | | | | |

### 1.2 Block 1 噪声候选清单

```
[已加噪 wffn2] ffn.output_proj                  → 操作数：gelu_out / W_ffn2 / b_ffn2
[未加噪]       ffn.layernorm.head.mean_smul     → 操作数：sum_x / 1/D
[未加噪]       ffn.layernorm.head.center_ctpt   → 操作数：x / square_mask
[未加噪]       ffn.layernorm.head.square_ctct   → 操作数：x_centered (self-mul)
[未加噪]       ffn.layernorm.head.var_smul      → 操作数：sum_xc2 / 1/D
```

---

## 2. Block 2 ＝ post-FFN LN tail + Wq/Wk(/Wv) + Q·K^T（`N=16384`）

> 用户手绘 Block2 计算图。**关键约束**：Q/K 共享 scaling factor。

### 2.1 计算流

| # | BLB 算子 | 输入 1 | 输入 2 | 输出 | 项目 id |
|---|---|---|---|---|---|
| 1 | **`ct*ct` = ewmulcc** | `x − mean : 30` (ct) | `1/std : 30` (ct) | (x−μ)/std | **`ffn.layernorm.tail.normalize_ctct`** |
| 2 | **`ct*pt` = ewmulcp** (× γ) | (x−μ)/std | `gama1 : 20` (pt) | γ·(x−μ)/std | **`ffn.layernorm.tail.scale_ctpt`** |
| 3 | Bs Rotation | normalized | — | — | — |
| 4 | **`ct*pt` = matmulcp** (Wq, Wk 同动作) | rotated | `Wq / Wk : 20` (pt) | K（同时也产 Q 的密文） | **`attn.q_proj`** ✅ `wq` & **`attn.k_proj`** ✅ `wk`（**shared_with**） |
| 5 | Gs Rotation | K | — | — | — |
| 6 | **`ct*pt` = matmulcp** (mask1) | rot K | `step1-baby step mask1 : 15` (pt) | rotK^T_mask1 | (BLB 协议步骤；PyTorch 无对应 id) |
| 7 | Bs Rotation step1 | | | | — |
| 8 | **`ct*pt` = matmulcp** (mask2) | | `step1-giant step mask2 : 15` (pt) | rotK^T | (BLB 协议步骤；PyTorch 无对应 id) |
| 9 | Gs Rotation step1 | | | | — |
| 10 | **`ct*ct` = matmulcc** (preprocess QK^T) | `ct*ct : Q` (ct) **🚧 TBD** | rotK^T | preprocessed QK^T | **`attn.qk_matmul`** |
| 11 | Gs Rotation step3 | | | | — |
| 12 | **`ct*pt` = ewmulcp** | preprocessed | `mask : 15` (pt) | **Q·K^T** | (BLB 协议步骤；可视为 attn.qk_matmul 的收尾) |
| → | 在 softmax 前还有 `attn.qk_scale_div` (× 1/√Dh，标量) | | | | **`attn.qk_scale_div`** |

> **🚧 V (`attn.v_proj`) 的精确融合 block 待用户补完图后确认**；当前注册表里默认放 Block 2，
> 等你下次告诉我 V 是在 Block 1 还是 Block 2 / 走哪条 ct*ct 路径，我立刻校正。

### 2.2 Block 2 噪声候选清单

```
[未加噪]       ffn.layernorm.tail.normalize_ctct   → x_centered / 1/std
[未加噪]       ffn.layernorm.tail.scale_ctpt        → normalized / γ_ffn
[已加噪 wq]    attn.q_proj                          → X / W_q / b_q     ┐ shared_with
[已加噪 wk]    attn.k_proj                          → X / W_k / b_k     ┘ (动作选择必须一致)
[已加噪 wv]    attn.v_proj                          → X / W_v / b_v     (🚧 block 归属 TBD)
[未加噪]       attn.qk_matmul                       → Q / K^T
[未加噪]       attn.qk_scale_div                    → attention_scores / 1/√Dh
```

---

## 3. Block 3 ＝ Softmax exp 近似主体（degree-aware）

> 用户手绘 Block3 计算图：每个 softmax degree 一张子图

我们项目里 softmax 走 `BertSelfAttentionWithAproximation.approximation_softmax`：

```
x = x − x.max(dim=-1) + 1e-9                   # 数值稳定
exp_approx = (1 + x / 2^degree) ** (2^degree)   # ←★ 这里是 Block 3 主体
exp_out = where(x < lower_bound, 0, exp_approx)
softmax = exp_out / (sum_exp + 1e-9)            # rec 非线性
```

**`degree` ∈ {2, 3, 4, 5, 6}** 决定 exp 近似中 `ct*ct` 自乘的次数：
- 入口：1 个 `ct*pt`，把 X 算成 `1 + x/(2^n)`（plaintext `1/(2^n)` scale=15）
- 然后 **`degree` 次** `ct*ct` 自乘 → `(1+x/2^n)^(2^degree) ≈ exp(x)`

### 3.1 各 degree 激活的 power 步骤

| degree | 用 N | scalar_div | power.s1 | s2 | s3 | s4 | s5 | s6 | norm_div |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2 | 8192  | ✓ | ✓ | ✓ | × | × | × | × | ✓ |
| 3 | 16384 | ✓ | ✓ | ✓ | ✓ | × | × | × | ✓ |
| 4 | 16384 | ✓ | ✓ | ✓ | ✓ | ✓ | × | × | ✓ |
| 5 | 16384 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × | ✓ |
| 6 | 16384 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

代码里调用 `select(blb_block=3, degree=D)` 直接拿到当前 degree 激活的所有 id。

### 3.2 各步骤项目 id

| BLB 步骤 | 项目 id | 操作数 |
|---|---|---|
| `ct*pt: 1 + x/(2^n)` | **`attn.softmax.scalar_div`** | `x_shifted` / `1/(2^n)` |
| `ct*ct: y → y^2` (第 1 次) | **`attn.softmax.power.s1`** | self-mul |
| `ct*ct: ^(2^2)` | **`attn.softmax.power.s2`** | self-mul |
| `ct*ct: ^(2^3)` | **`attn.softmax.power.s3`** | self-mul |
| `ct*ct: ^(2^4)` | **`attn.softmax.power.s4`** | self-mul |
| `ct*ct: ^(2^5)` | **`attn.softmax.power.s5`** | self-mul |
| `ct*ct: ^(2^6)` | **`attn.softmax.power.s6`** | self-mul |
| `exp_out / sum_exp`（rec 非线性 + smul） | **`attn.softmax.norm_div`** | `exp_out` / `1/(sum_exp+1e-9)` |

### 3.3 Block 3 噪声候选清单

```
所有 degree 共有
[未加噪] attn.softmax.scalar_div    → x_shifted / 1/(2^n)
[未加噪] attn.softmax.norm_div      → exp_out / 1/(sum_exp+1e-9)

按 degree 激活的 power 步骤
softmax2:  power.s1, power.s2                                         (用 N=8192)
softmax3:  power.s1..s3                                               (用 N=16384)
softmax4:  power.s1..s4                                               (用 N=16384)
softmax5:  power.s1..s5                                               (用 N=16384)
softmax6:  power.s1..s6                                               (用 N=16384)

[未加噪] attn.head_mask_mul          → probs / head_mask  (默认 None 不触发)
```

---

## 4. Block 4 ＝ rot_softmax · V (双起点 CTCT_MUL) + Wo + post-attn LN head（`N=16384`）

> 用户手绘 Block4 计算图。**关键特性**：rot_softmax · V 是双起点的 CTCT_MUL，
> 主链(rot_softmax) 和副链(V) 可独立选 scaling factor / 噪声。

### 4.1 计算流

| # | BLB 算子 | 输入 1 (主链) | 输入 2 | 输出 | 项目 id |
|---|---|---|---|---|---|
| 1a | **`ct*pt` = ewmulcp** (主链 mask 2) | `rot softmax` (ct) | `step1-giant step mask2 : 15` | masked rot_softmax | (BLB 协议步骤) |
| 1b | **`ct*pt` = ewmulcp** (副链 mask 2) | `V` (ct) **🚧 TBD** | `step1-giant step mask2 : 15` | masked V | (BLB 协议步骤) |
| 2a | Rot Gs Step2 (主链) | | | | — |
| 2b | Rot Bs Step2 (副链) | | | | — |
| 3 | **`ct*ct` = matmulcc** **CTCT_MUL** | masked rot_softmax (主链) | masked V (副链) | rot_softmax · V (preliminary) | **`attn.probs_v_matmul`** ✅ `softmax_probs / value_after_softmax` (双独立噪声) |
| 4 | Rot St3 | | | | — |
| 5 | **`ct*pt` = ewmulcp** | | `mask : 15` | **softmax · V** | (协议收尾) |
| 6 | Rot Ct*Wo | | | | — |
| 7 | **`ct*pt` = matmulcp** (Wo) | softmax · V | `Wo : 20` | **AttnOut** | **`attn.o_proj`** ✅ `wo` |
| 8 | Rotation 3 次 | | | | — |
| 9 | **`ct*pt` = smulcp** (× 1/D) | sum | `1/D : 20` | mean | **`attn.layernorm.head.mean_smul`** |
| 10 | **`ct*pt` = ewmulcp** (square mask) | x | square mask | (BLB 协议) | **`attn.layernorm.head.center_ctpt`** |
| 11 | **`ct*ct` = ewmulcc** | x_centered | x_centered | (x − μ)² | **`attn.layernorm.head.square_ctct`** |
| 12 | Rotation 3 次 | | | | — |
| 13 | **`ct*pt` = smulcp** (× 1/D) | sum_xc2 | `1/D : 20` | **(x−μ)²·1/D = variance** | **`attn.layernorm.head.var_smul`** |
| → | (rsqrt 走 MPC，输出 1/std 进入 Block 5) | | | | |

> **关于 `attn.probs_v_matmul`**：BLB rescale skeleton 字段
> `ctct_rot_softmax_mul_v.scale_delta_bits=0` / `other_ct_scale_bits=39` 表示
> 主链 (rot_softmax) 和副链 (V) 在 ct*ct 处独立计噪。我们现行实现已经在
> `_apply_softmax_value_noise()` 里给两个操作数分别用了 `softmax_scaling_factor`
> / `value_scaling_factor`，**正好对应**。

### 4.2 Block 4 噪声候选清单

```
[未加噪]                                attn.head_mask_mul              → probs / head_mask (默认不触发)
[已加噪 softmax_probs/value_after_softmax] attn.probs_v_matmul          → 主链 rot_softmax / 副链 V
[已加噪 wo]                             attn.o_proj                     → context / W_o / b_o
[未加噪]                                attn.layernorm.head.mean_smul   → sum_x / 1/D
[未加噪]                                attn.layernorm.head.center_ctpt → x / square_mask
[未加噪]                                attn.layernorm.head.square_ctct → x_centered (self-mul)
[未加噪]                                attn.layernorm.head.var_smul    → sum_xc2 / 1/D
```

---

## 5. Block 5 ＝ post-attn LN tail + Wffn1 + GeLU 多项式（degree-aware）

> 用户手绘 Block5 计算图。**关键约束**：gelu4 的两个 ct*ct 共享 scaling factor。

我们项目里 GELU 走 `PolynomialGELU`：

```python
def polynomial(x, coeff, sign):
    powers = torch.stack([x.pow(i) for i in range(len(coeff[sign]))], dim=-1)
    return (powers * coeff_tensor).sum(dim=-1)
```

**`degree` ∈ {1, 2, 4}** 决定多项式次数：
- degree 1：`y = a + b·x`
- degree 2：`y = a + b·x + c·x²`
- degree 4：`y = a + b·x + c·x² + d·x³ + e·x⁴`

### 5.1 公共前段（所有 degree 共用）

| # | BLB 算子 | 输入 1 | 输入 2 | 项目 id |
|---|---|---|---|---|
| 1 | **`ct*ct` = ewmulcc** (normalize) | `x − mean` (ct) | `1/std` (ct) | **`attn.layernorm.tail.normalize_ctct`** |
| 2 | **`ct*pt` = ewmulcp** (× γ) | normalized | `gama1 : 20` (pt) | **`attn.layernorm.tail.scale_ctpt`** |
| 3 | Bs Rotation | | | — |
| 4 | **`ct*pt` = matmulcp** (Wffn1) | rotated | `Wffn1 : 22` (pt) | **`ffn.intermediate_proj`** ✅ `wffn1` |
| 5 | Gs Rotation → `ct : ffn1_out` | | | — |

### 5.2 各 degree 的 GeLU 多项式段

| degree | 用 N | power.x2 | power.x3x4 | coeff_mul (final ct*pt) |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 8192  | × | × | ✓（系数 a, b） |
| 2 | 16384 | ✓ | × | ✓（系数 a, b, c） |
| 4 | 16384 | ✓ | ✓ | ✓（系数 a, b, c, d, e） |

> **gelu4 约束**：`ffn.gelu.power.x2` 与 `ffn.gelu.power.x3x4` 必须共享 scaling factor，
> 即"选相同 scaling factor，加相同噪音"。（注册表 shared_with 已记录。）

| BLB 步骤 | 项目 id | 触发 degree |
|---|---|---|
| `ct*ct: x · x = x²` | **`ffn.gelu.power.x2`** | 2, 4 |
| `ct*ct: x²·x²/x²·x` (并行算 x⁴ 与 x³) | **`ffn.gelu.power.x3x4`** | 4 |
| `ct*pt: Σ coeff_i · x^i` (最终多项式合成) | **`ffn.gelu.coeff_mul`** | 1, 2, 4 |

### 5.3 Block 5 噪声候选清单

```
所有 degree 共有
[未加噪]      attn.layernorm.tail.normalize_ctct   → x_centered / 1/std
[未加噪]      attn.layernorm.tail.scale_ctpt        → normalized / γ_attn
[已加噪 wffn1] ffn.intermediate_proj                → X / W_ffn1 / b_ffn1

按 degree 激活的 GeLU 步骤
gelu1:  ffn.gelu.coeff_mul                          (用 N=8192;  系数 a, b)
gelu2:  ffn.gelu.power.x2, ffn.gelu.coeff_mul       (用 N=16384; 系数 a, b, c)
gelu4:  ffn.gelu.power.x2, ffn.gelu.power.x3x4,
        ffn.gelu.coeff_mul                          (用 N=16384; 系数 a..e；
                                                     两个 power 步必须共享 sf)
```

---

## 6. Block 之外：Embeddings / Pooler / Classifier（一次性）

不在 BLB Figure 10 的 5-block 循环内。注册表里 `blb_block=="embeddings"/"pooler"/"head"`，
`blb_N=None`。

### 6.1 Stage 0：BertEmbeddings

| 项目 id | op_type | 已加噪？ |
|---|---|---|
| `emb.word_lookup` | embedding_lookup | ❌ |
| `emb.token_type_lookup` | embedding_lookup | ❌ |
| `emb.position_lookup` | embedding_lookup | ❌ |
| `emb.layernorm.head.mean_smul` | scalar_mul | ❌ |
| `emb.layernorm.head.center_ctpt` | elementwise_mul | ❌ |
| `emb.layernorm.head.square_ctct` | elementwise_mul | ❌ |
| `emb.layernorm.head.var_smul` | scalar_mul | ❌ |
| `emb.layernorm.tail.normalize_ctct` | elementwise_mul | ❌ |
| `emb.layernorm.tail.scale_ctpt` | elementwise_mul | ❌ |

### 6.2 Stage 3 / 4：Pooler + Classifier

| 项目 id | op_type | 已加噪？ |
|---|---|---|
| `pooler.dense` | linear_mm | ❌ |
| `head.classifier` | linear_mm | ❌ |

---

## 7. 全部 44 个 id 速查（按 BLB 时间序）

```
─── Embeddings (一次性) ────────────────────────────────────────
emb.word_lookup
emb.token_type_lookup
emb.position_lookup
emb.layernorm.head.mean_smul      ┐
emb.layernorm.head.center_ctpt    │  Embedding LayerNorm
emb.layernorm.head.square_ctct    │  (在 BLB 5-block 循环之外)
emb.layernorm.head.var_smul       │
emb.layernorm.tail.normalize_ctct │
emb.layernorm.tail.scale_ctpt     ┘

─── Per-layer 5-block 循环 ────────────────────────────────────
═ Block 1 (Wffn2 + post-FFN LN head; N=8192) ═
ffn.output_proj                            [✅ wffn2]
ffn.layernorm.head.mean_smul
ffn.layernorm.head.center_ctpt
ffn.layernorm.head.square_ctct
ffn.layernorm.head.var_smul

═ Block 2 (post-FFN LN tail + Q/K/V + Q·K^T; N=16384) ═
ffn.layernorm.tail.normalize_ctct
ffn.layernorm.tail.scale_ctpt
attn.q_proj                                [✅ wq]   ┐ shared_with
attn.k_proj                                [✅ wk]   ┘ (Q/K 一致)
attn.v_proj                                [✅ wv]   (🚧 block 归属 TBD)
attn.qk_matmul
attn.qk_scale_div

═ Block 3 (Softmax exp 近似; degree-aware) ═
attn.softmax.scalar_div                    (degree ∈ {2,3,4,5,6})
attn.softmax.power.s1                      (degree ≥ 2)
attn.softmax.power.s2                      (degree ≥ 2)
attn.softmax.power.s3                      (degree ≥ 3, N=16384)
attn.softmax.power.s4                      (degree ≥ 4, N=16384)
attn.softmax.power.s5                      (degree ≥ 5, N=16384)
attn.softmax.power.s6                      (degree = 6, N=16384)
attn.softmax.norm_div                      (degree ∈ {2,3,4,5,6})

═ Block 4 (probs·V + Wo + post-attn LN head; N=16384) ═
attn.head_mask_mul                         (默认不触发)
attn.probs_v_matmul                        [✅ softmax_probs / value_after_softmax]
attn.o_proj                                [✅ wo]
attn.layernorm.head.mean_smul
attn.layernorm.head.center_ctpt
attn.layernorm.head.square_ctct
attn.layernorm.head.var_smul

═ Block 5 (post-attn LN tail + Wffn1 + GeLU; degree-aware) ═
attn.layernorm.tail.normalize_ctct
attn.layernorm.tail.scale_ctpt
ffn.intermediate_proj                      [✅ wffn1]
ffn.gelu.power.x2                          (degree ∈ {2,4}, N=16384)  ┐ shared_with
ffn.gelu.power.x3x4                        (degree = 4,    N=16384)   ┘ (gelu4 内一致)
ffn.gelu.coeff_mul                         (degree ∈ {1,2,4})

─── Pooler / Classifier (一次性) ──────────────────────────────
pooler.dense
head.classifier
```

---

## 8. 当前已加噪声的 id

| 噪声键 | 注入点（`function_handler.py`） | 影响的 id（按 BLB block） |
|---|---|---|
| `'x'` | `BertLayer.forward` 入口包装 | Block 2 的 `attn.q_proj` / `attn.k_proj` / `attn.v_proj` 的 X 操作数 |
| `'wq'` | `attention.self.query.forward` | Block 2 `attn.q_proj` 的 W |
| `'wk'` | `attention.self.key.forward` | Block 2 `attn.k_proj` 的 W |
| `'wv'` | `attention.self.value.forward` | Block 2 `attn.v_proj` 的 W |
| `'wo'` | `attention.output.dense.forward` | Block 4 `attn.o_proj` 的 W |
| `'wffn1'` | `intermediate.dense.forward` | Block 5 `ffn.intermediate_proj` 的 W |
| `'wffn2'` | `output.dense.forward` | Block 1 `ffn.output_proj` 的 W |
| `'softmax_probs'` | `_apply_softmax_value_noise` | Block 4 `attn.probs_v_matmul` 的 主链 (rot_softmax) 操作数 |
| `'value_after_softmax'` | `_apply_softmax_value_noise` | Block 4 `attn.probs_v_matmul` 的 副链 (V) 操作数 |

可用的 scaling factor 集合（legacy 表）：
```
INPUT_NOISE_ALLOWED_SCALING_FACTORS  = (22, 24, 26, 28, 30)
WEIGHT_NOISE_ALLOWED_SCALING_FACTORS = (14, 16, 18, 20, 22)
WFFN1_NOISE_ALLOWED_SCALING_FACTORS  = (16, 18, 20, 22, 24)
SOFTMAX_VALUE_NOISE_ALLOWED_SCALING_FACTORS = (10..46)
```

新（多 N）表的 scaling factor 集合：
```
NOISE_VARIANCE_TABLE_BY_N[N] 支持 scale_bits ∈ [10, 46]，N ∈ {8192, 16384}
```

---

## 9. 还没加噪声的 id（按 BLB block 分组，方便点名）

> 报噪声指令格式：
> ```
> id = <id 名>，操作数 = <X / W / γ / 1/D / 1/(2^n) / coeff / mask / ...>，
> scaling factor = <K>，分布 = <encoding/fresh/rescale>，
> 用 N=<8192/16384>
> [条件：仅当 softmax degree = D / GELU degree = D 时]
> ```

### Block 1（N=8192）
- `ffn.output_proj` 的 X (= GELU_out)
- `ffn.layernorm.head.{mean_smul, center_ctpt, square_ctct, var_smul}` 全部 4 条

### Block 2（N=16384）
- `ffn.layernorm.tail.normalize_ctct` 的 (x−mean) / (1/std)
- `ffn.layernorm.tail.scale_ctpt` 的 normalized / γ
- `attn.q_proj` / `attn.k_proj` / `attn.v_proj` 的 **X 操作数**（W 已加噪；Q/K 必须共享）
- `attn.qk_matmul` 的 Q / K^T
- `attn.qk_scale_div` 的 attention_scores / 1/√Dh

### Block 3（degree 决定）
- `attn.softmax.scalar_div`（所有 degree；plaintext 是 1/(2^n) at scale=15）
- `attn.softmax.power.s1` ~ `s6`（按当前 degree 截断；s1/s2 用 N=8192 仅当 degree=2，否则 N=16384）
- `attn.softmax.norm_div` exp_out
- `attn.head_mask_mul`（默认不触发）

### Block 4（N=16384）
- `attn.probs_v_matmul`：当前已加噪；如要重新调主/副链 sf，可重指定
- `attn.o_proj` 的 X (= softmax · V)
- `attn.layernorm.head.{mean_smul, center_ctpt, square_ctct, var_smul}` 全部 4 条

### Block 5（gelu degree 决定）
- `attn.layernorm.tail.normalize_ctct` / `attn.layernorm.tail.scale_ctpt`
- `ffn.intermediate_proj` 的 X
- `ffn.gelu.power.x2`（degree ∈ {2,4}）
- `ffn.gelu.power.x3x4`（degree = 4；与 x2 共享 sf）
- `ffn.gelu.coeff_mul`（所有 degree；系数数量随 degree 变化）

### Embeddings / Pooler / Head
- `emb.*` 9 条全部
- `pooler.dense`、`head.classifier`

---

## 10. 协作约定

1. **新增噪声候选**：先在 [`noise_targets_registry.py`](../noise_targets_registry.py) 的
   `NOISE_TARGETS` 末尾追加 dict，再到本 md 第 1–6 节相应 block 表里补一行。
2. **报噪声指令**：用第 9 节的格式，按 id 精确点名。批量指令可以直接给我一个 id 列表，
   我会按 `select(...)` + 注册表 metadata 自动算 N、检查 shared_with 约束。
3. **共享 scaling factor 约束**：`shared_groups()` 返回所有约束组。当前是
   `[('attn.k_proj', 'attn.q_proj'), ('ffn.gelu.power.x2', 'ffn.gelu.power.x3x4')]`。
   后面写注入器时会强制这两组里的 id 用同一个 scaling factor。
4. **🚧 仍 TBD**：
   - Block 2 `ct*ct: Q` 的精确生成步骤（用户下次补图）
   - Block 4 副链 `V` 怎么从更早 block 流过来（用户下次补图）
   等你下次发我新图，我就更新本 md 和注册表的 `attn.v_proj.blb_block` / `attn.qk_matmul.notes`。
5. **多 N 噪声方差表**：通过 `function_handler.get_input_noise_variance_by_N(scale, dist, N)`
   获取 σ²（注意是 std² 不是 std）。Legacy `INPUT_NOISE_VARIANCE_TABLE` 仍可用，
   走 `get_input_noise_variance(scale, dist)`。
