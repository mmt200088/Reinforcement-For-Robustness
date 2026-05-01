# 噪声注入候选清单 —— 按 BLB Figure 10 的 5-block 划分梳理

> 真相来源：
> 1. **BLB 论文 Figure 10**（Tianshi Xu, Wen-jie Lu et al., *"Breaking the Layer Barrier:
>    Remodeling Private Transformer Inference with Hybrid CKKS and MPC"*, USENIX Security 2025,
>    arXiv [2508.19525](https://arxiv.org/abs/2508.19525)）
> 2. **本项目用户手绘的 Block1–Block5 计算图**（与 Figure 10 一一对应、并标注了
>    每个 block 用哪一张噪声方差表 N=8192 / N=16384）
> 3. **代码真相**：[`function_handler.py`](../function_handler.py)
>    （`PolynomialGELU` / `BertSelfAttentionWithAproximation.approximation_softmax`）
> 4. **本项目噪声候选注册表**：[`noise_targets_registry.py`](../noise_targets_registry.py)
>
> 你后面要说 *"在哪些 id 上加噪声、怎么加"* 时，直接报本文里的 `id` 即可。

---

## 0. 全局约定

### 0.1 BLB 算子分类（论文 Table 2）

| 类别 | 维度变换 | 代表算子（CKKS） |
|---|---|---|
| **Identity** | (L,D) → (L,D) | `ewadd`、`ewmul` |
| **Expansion** | (L,1) → (L,D) | `sadd`、`smul` |
| **Reduction** | (L,D) → (L,1) | `sum` |
| **Transformation** | (L,D) → (L,D′) | `matmul`（`matmulcc`、`matmulcp`） |
| 非线性 | — | `cmp`、`mux`、`rec`（求倒数）、`rsqrt` |

记号：`ct*ct` = `matmulcc / ewmulcc`（密-密乘）；`ct*pt` = `matmulcp / ewmulcp`（密-明乘）。

### 0.2 BERT 架构

**BLB / 本项目都用 post-LN**：每个 BertLayer 内部走的是
"Attention → +residual → LayerNorm → FFN → +residual → LayerNorm" 的顺序。

### 0.3 BLB Figure 10 的 5-block 切分

BLB 把一个 Transformer layer **的所有线性算子** 跨 LayerNorm / Softmax / GeLU 等
非线性边界，重新切分成 5 个可融合的 block。Block 之间的边界是非线性算子
（`rsqrt` / `cmp` / `mux` / `rec`），它们在 BLB 框架里走 MPC，不属于任何一个 block。

| Block | 颜色（图例） | 在 layer 中的内容 | 用哪张噪声表（用户标注） |
|---|---|---|---|
| **Block 1** | 绿 | **Wffn2 投影** + LayerNorm 头部（计算 mean、(x-mean)²·1/D = variance） | `N=8192` |
| **Block 2** | 蓝 | **LayerNorm 尾部**（×1/std、×γ） + **Wq、Wk、Wv 三个投影** + **Q·K^T** | `N=16384` |
| **Block 3** | 粉 | **Softmax 的 exp 近似主体**（degree 决定步数） | softmax2 → `N=8192`；softmax3/4/5/6 → `N=16384` |
| **Block 4** | 紫 | **softmax · V** + **Wo 投影** + LayerNorm 头部（mean、variance） | `N=16384` |
| **Block 5** | 青 | **LayerNorm 尾部**（×1/std、×γ） + **Wffn1 投影** + **GeLU 多项式**（degree 决定步数） | gelu1 → `N=8192`；gelu2/gelu4 → `N=16384` |

> 时间序：Block 5 → (rsqrt 非线性) → Block 1 → (rsqrt 非线性) → Block 2 →
> (cmp/mux/rec 非线性，Block 3 内部) → Block 4 → (rsqrt 非线性) → Block 5（下一层）。
> Block 1、Block 4 的"LayerNorm 头部"和 Block 2、Block 5 的"LayerNorm 尾部"
> 在物理上属于 **同一个** LayerNorm（中间隔了一个 `rsqrt`），但 BLB 把它们切到
> 不同 block 里以最大化跨非线性的线性融合。

### 0.4 用户手绘 block 计算图里的两个 TBD

用户在两张图上明确写了 *"先别加，要补充一下"*：

1. **Block 2 里的 `ct*ct: Q`**（preprocess QK^T 的 Q 操作数）
   —— Q 投影本身的精确步骤待你补充。
2. **Block 4 里的 `V`**（softmax · V 的 V 操作数）
   —— V 怎么从更早的 block 流到这里待你补充。

下面表格里凡是涉及到这两处的位置都标 **🚧 TBD**。

---

## 1. Block 1 ＝ Wffn2 + LayerNorm 头部 (mean, variance)

> 用户手绘 Block1 计算图（N=8192 噪声表）

### 1.1 计算流（按时间顺序）

| # | 算子（BLB / 用户标记） | 输入 1 | 输入 2 | 输出 |
|---|---|---|---|---|
| 1 | Bs Rotation in mul | `GELU_out` (ct) | — | rotated ct |
| 2 | **`ct*pt` = matmulcp**（**Wffn2**） | rotated ct | `Wffn2 : 20` (pt) | ffn2_out (ct) |
| 3 | Gs Rotation in mul | ffn2_out | — | rotated |
| 4 | **Rotation Sum1: 3 次** = `sum` | rotated | — | sum (用于 mean) |
| 5 | **`ct*pt` = smulcp**（× 1/D） | sum | `1/D : 20` (pt) | **mean** (ct) |
| 6 | **`ct*pt` = ewmulcp** 或 `ct*ct` 计算 (x − mean) 然后平方 | ffn2_out | `result of ct*pt (square)` | (x − mean)² (ct) |
| 7 | **`ct*ct` = ewmulcc** 平方 | (x−mean) | (x−mean) | (x−mean)² |
| 8 | **Rotation Sum2: 3 次** = `sum` | (x−mean)² | — | sum |
| 9 | **`ct*pt` = smulcp**（× 1/D） | sum | `1/D : 20` (pt) | **(x−mean)²·1/D = variance** |
| → | (后续 `rsqrt` 走 MPC，输出 1/std 进入 Block 2) | | | |

### 1.2 项目里对应的注册表 id

| BLB 步骤 | 项目 id | op_type | 已加噪？ |
|---|---|---|---|
| 步骤 2 (Wffn2 投影) | `ffn.output_proj` | linear_mm | ✅ `wffn2`（噪在 W_ffn2 上） |
| 步骤 5、9 (× 1/D 求 mean / variance) | `ffn.layernorm.stat_div`（聚合表示） | stat_div | ❌ |
| 步骤 7 ((x−mean)² 平方) | `ffn.layernorm.stat_div`（聚合表示） | stat_div | ❌ |

> 备注：`ffn.layernorm.stat_div` 这一个 id 在我们当前注册表里**聚合表示**了"LN 头部到
> rsqrt 之前"的所有乘法（mean 的 1/D、variance 的 (x-mean)² 与 1/D）。如果你后面想
> 单独控制其中某一步（比如只对 (x-mean)² 加噪声），告诉我，我会在注册表里把它拆成
> 三个细粒度子 id：`ffn.layernorm.head.mean_norm` / `ffn.layernorm.head.square` /
> `ffn.layernorm.head.var_norm`。

### 1.3 Block 1 总噪声候选清单

```
[已加噪] ffn.output_proj                         → W_ffn2 操作数
[未加噪] ffn.output_proj                         → 输入 X (= GELU_out) 操作数
[未加噪] ffn.layernorm.stat_div  (mean 步)        → ffn2_out 操作数 / 1/D 操作数
[未加噪] ffn.layernorm.stat_div  (square 步)      → (x-mean) 两个操作数（self-mul）
[未加噪] ffn.layernorm.stat_div  (variance 步)    → (x-mean)² 操作数 / 1/D 操作数
```

---

## 2. Block 2 ＝ LayerNorm 尾部 + Wq/Wk(/Wv) + Q·K^T

> 用户手绘 Block2 计算图（N=16384 噪声表）

### 2.1 计算流

| # | 算子 | 输入 1 | 输入 2 | 输出 |
|---|---|---|---|---|
| 1 | **`ct*ct` = ewmulcc**（normalize） | `x − mean : 30` (ct, 来自 Block 1) | `1/std : 30` (ct, rsqrt 输出) | (x−mean)/std |
| 2 | **`ct*pt` = ewmulcp**（× γ） | (x−mean)/std | `gama1 : 20` (pt) | γ·(x−mean)/std |
| 3 | Bs Rotation | normalized | — | rotated |
| 4 | **`ct*pt` = matmulcp**（**Wq / Wk**） | rotated | `Wq / Wk : 20` (pt) | K（或 Q） |
| 5 | Gs Rotation | K | — | rotated K |
| 6 | **`ct*pt` = matmulcp** (mask1) | rotated K | `step1-baby step mask1 : 15` (pt) | rotK^T_mask1 |
| 7 | Bs Rotation step1 | | | |
| 8 | **`ct*pt` = matmulcp** (mask2) | | `step1-giant step mask2 : 15` (pt) | rotK^T |
| 9 | Gs Rotation step1 | | | |
| 10 | **`ct*ct` = matmulcc** preprocess QK^T | `ct*ct : Q` (ct) **🚧 TBD** | rotK^T | preprocessed QK^T |
| 11 | Gs Rotation step3 | | | |
| 12 | **`ct*pt` = ewmulcp** | preprocessed | `mask : 15` (pt) | **Q·K^T** |
| → | (Block 3 接 softmax) | | | |

> **🚧 TBD**：Q 投影 (`Wq`) 的精确融合步骤、以及 Wv 在哪个 block 计算，都是用户标注
> 的"先别加，要补充"区域。当前最稳的猜测是 Wq、Wk、Wv 三个 ct*pt 都在 Block 2，
> 与 BLB Figure 10 三个 matmul 框（V/Q/K）颜色一致——但具体顺序需要你补完图后再定。

### 2.2 项目里对应的注册表 id

| BLB 步骤 | 项目 id | op_type | 已加噪？ |
|---|---|---|---|
| 步骤 1 (× 1/std) | `attn.layernorm.stat_div` | stat_div | ❌ |
| 步骤 2 (× γ) | `attn.layernorm.scale_mul` | elementwise_mul | ❌ |
| 步骤 4 (Wq) | `attn.q_proj` | linear_mm | ✅ `wq` |
| 步骤 4' (Wk) | `attn.k_proj` | linear_mm | ✅ `wk` |
| 步骤 4'' (Wv，TBD 位置) | `attn.v_proj` | linear_mm | ✅ `wv` |
| 步骤 10 (Q · K^T) | `attn.qk_matmul` | activation_mm | ❌ |
| 步骤 12 中的 `1/√Dh` 缩放（论文里出现位置不同，我们项目在 softmax 前显式做） | `attn.qk_scale_div` | scalar_mul | ❌ |

### 2.3 Block 2 总噪声候选清单

```
[未加噪] attn.layernorm.stat_div    → (x-mean) 操作数 / (1/std) 操作数
[未加噪] attn.layernorm.scale_mul   → normalized 操作数 / γ 操作数
[已加噪] attn.q_proj                 → W_q 操作数  (X 操作数未加)
[已加噪] attn.k_proj                 → W_k 操作数  (X 操作数未加)
[已加噪] attn.v_proj                 → W_v 操作数  (X 操作数未加)
[未加噪] attn.qk_matmul              → Q 操作数 / K^T 操作数
[未加噪] attn.qk_scale_div           → attention_scores 操作数 / (1/√Dh) 操作数
```

---

## 3. Block 3 ＝ Softmax 的 exp 近似主体（按 degree 展开）

> 用户手绘 Block3 计算图：每个 softmax degree 一张子图

我们项目里 softmax 走 `BertSelfAttentionWithAproximation.approximation_softmax`：

```
x = x − x.max(dim=-1) + 1e-9                # 数值稳定（addition only）
exp_approx = (1 + x / 2^degree) ** (2^degree)
exp_out = where(x < lower_bound, 0, exp_approx)
softmax = exp_out / (sum_exp + 1e-9)        # rec 是非线性
```

**`degree`（用户允许选 2、3、4、5、6）** 决定 exp 近似里需要多少次 ct*ct 平方：
- 输入 X 经过 `ct*pt = (1 + x/2^n)` 一次（`x : 30`，`1/(2^n) : 15`）
- 然后做 **`degree` 次** `ct*ct` 平方，得到 `(1+x/2^n)^(2^degree) ≈ exp(x)`

### 3.1 各 degree 的算子序列

#### softmax2（degree = 2，**N = 8192**）

| # | 算子 | 输入 1 | 输入 2 | 输出 |
|---|---|---|---|---|
| 1 | `ct*pt` | `X : 30` | `1/(2^n) : 15` | `1 + x/(2^n)` |
| 2 | `ct*ct` 平方 | 上 | 上 (self-mul) | `(1+x/(2^n))^2` |
| 3 | `ct*ct` 平方 | 上 | 上 | `(1+x/(2^n))^(2^2) ≈ exp(x)` |

#### softmax3（degree = 3，**N = 16384**）

| # | 算子 | 输入 1 | 输入 2 | 输出 |
|---|---|---|---|---|
| 1 | `ct*pt` | `X : 30` | `1/(2^n) : 15` | `1 + x/(2^n)` |
| 2 | `ct*ct` | 上 | 上 | `^2` |
| 3 | `ct*ct` | 上 | 上 | `^(2^2)` |
| 4 | `ct*ct` | 上 | 上 | `^(2^3) ≈ exp` |

#### softmax4（degree = 4，**N = 16384**）

| # | 算子 | 输入 1 | 输入 2 | 输出 |
|---|---|---|---|---|
| 1 | `ct*pt` | `X : 30` | `1/(2^n) : 15` | `1 + x/(2^n)` |
| 2–5 | `ct*ct` × 4 | self-mul × 4 次 | | `^(2^4) ≈ exp` |

#### softmax5（degree = 5，**N = 16384**）

| # | 算子 |
|---|---|
| 1 | `ct*pt`  `(1 + x/(2^n))` |
| 2–6 | `ct*ct` × 5 → `^(2^5)` ≈ exp |

#### softmax6（degree = 6，**N = 16384**）

| # | 算子 |
|---|---|
| 1 | `ct*pt`  `(1 + x/(2^n))` |
| 2–7 | `ct*ct` × 6 → `^(2^6)` ≈ exp |

### 3.2 exp 之后的 softmax 收尾（仍属于 Block 3 / 通往 Block 4 的边界）

```
exp_out = where(x < lower_bound, 0, exp_approx)   # cmp + mux 非线性
sum_exp = sum(exp_out, dim=-1)                    # sum reduction
softmax = exp_out / sum_exp                       # rec + smul
```

### 3.3 项目里对应的注册表 id（与 degree 无关，按算子聚合）

| BLB 步骤 | 项目 id | op_type | 已加噪？ |
|---|---|---|---|
| `ct*pt` 初始 (1 + x/2^n) | `attn.softmax.expapprox.scalar_div` | scalar_mul | ❌ |
| `ct*ct` × degree 次平方 | `attn.softmax.expapprox.power` | self_power | ❌ |
| 末端的 exp/sum_exp | `attn.softmax.norm_div` | stat_div | ❌ |

> ⚠️ 当前 `attn.softmax.expapprox.power` 是单一 id，**没有按 degree / step 拆细**。
> 如果你后面要"只对 softmax6 的第 4 次平方加噪声"这种粒度，告诉我，我会在注册表里
> 拆成 `attn.softmax.expapprox.power.s1` ~ `s6`（最多 6 步），或按 degree 维度
> `attn.softmax.expapprox.power.d{2,3,4,5,6}.s{1..d}`。

### 3.4 Block 3 总噪声候选清单（按 degree）

```
所有 degree 共有
[未加噪] attn.softmax.expapprox.scalar_div    → x_shifted 操作数 / 1/(2^n) 操作数
[未加噪] attn.softmax.norm_div                → exp_out 操作数 (rec 输入是非线性)

按 degree 区分平方步数
[未加噪] attn.softmax.expapprox.power · d=2 (用 softmax2 时)  →  step 1, 2
[未加噪] attn.softmax.expapprox.power · d=3 (用 softmax3 时)  →  step 1, 2, 3
[未加噪] attn.softmax.expapprox.power · d=4 (用 softmax4 时)  →  step 1, 2, 3, 4
[未加噪] attn.softmax.expapprox.power · d=5 (用 softmax5 时)  →  step 1, 2, 3, 4, 5
[未加噪] attn.softmax.expapprox.power · d=6 (用 softmax6 时)  →  step 1, 2, 3, 4, 5, 6

[未加噪] attn.head_mask_mul    (默认 head_mask=None 不触发)
```

---

## 4. Block 4 ＝ softmax · V + Wo + LayerNorm 头部

> 用户手绘 Block4 计算图（N=16384 噪声表）

### 4.1 计算流

| # | 算子 | 输入 1 | 输入 2 | 输出 |
|---|---|---|---|---|
| 1 | **`ct*pt` = ewmulcp** (mask 2) | `rot softmax` (ct) | `step1-giant step mask2 : 15` (pt) | masked softmax |
| 2 | Rot Gs Step2 | | | |
| 3 | **`ct*ct` = matmulcc** (rot softmax · v) | masked softmax | `V` (ct) **🚧 TBD** | softmax·V (preliminary) |
| 4 | Rot St3 | | | |
| 5 | **`ct*pt` = ewmulcp** (mask) | | `mask : 15` (pt) | **softmax · V** |
| 6 | Rot Ct\*Wo | | | |
| 7 | **`ct*pt` = matmulcp** (Wo) | softmax · V | `Wo : 20` (pt) | **AttnOut** |
| 8 | Rotation 3 次 | | | |
| 9 | **`ct*pt` = smulcp** (× 1/D) | | `1/D : 20` (pt) | mean |
| 10 | (减均值后) **`ct*ct` = ewmulcc** 平方 | (x−mean) | (x−mean) | (x−mean)² |
| 11 | Rotation 3 次 | | | |
| 12 | **`ct*pt` = smulcp** (× 1/D) | | `1/D : 20` (pt) | **(x−mean)²·1/D = variance** |
| → | (后续 rsqrt 走 MPC，输出进入下一个 Block 5) | | | |

### 4.2 项目里对应的注册表 id

| BLB 步骤 | 项目 id | op_type | 已加噪？ |
|---|---|---|---|
| 步骤 3 (softmax · V matmulcc) | `attn.probs_v_matmul` | activation_mm | ✅ `softmax_probs / value_after_softmax`（两个操作数都加 fresh 噪声） |
| 步骤 7 (Wo) | `attn.o_proj` | linear_mm | ✅ `wo`（噪在 W_o 上） |
| 步骤 9、12 (× 1/D 求 mean / variance) | `attn.layernorm.stat_div`（聚合） | stat_div | ❌ |
| 步骤 10 ((x−mean)² 平方) | `attn.layernorm.stat_div`（聚合） | stat_div | ❌ |

### 4.3 Block 4 总噪声候选清单

```
[已加噪] attn.probs_v_matmul             → probs / V 操作数（softmax_probs / value_after_softmax）
[已加噪] attn.o_proj                      → W_o 操作数  (X = softmax·V 操作数未加)
[未加噪] attn.layernorm.stat_div  (mean)
[未加噪] attn.layernorm.stat_div  (square)
[未加噪] attn.layernorm.stat_div  (variance)
```

---

## 5. Block 5 ＝ LayerNorm 尾部 + Wffn1 + GeLU 多项式（按 degree 展开）

> 用户手绘 Block5 计算图：每个 GELU degree 一张子图

我们项目里 GELU 走 `PolynomialGELU`：

```python
def polynomial(x, coeff, sign):
    powers = torch.stack([x.pow(i) for i in range(len(coeff[sign]))], dim=-1)
    return (powers * coeff_tensor).sum(dim=-1)
```

**`degree`（用户允许选 1、2、4）** 决定多项式次数：
- degree 1：`y = a + b·x`（线性，用一次 `ct*pt` 完成）
- degree 2：`y = a + b·x + c·x²`（先一次 `ct*ct` 算 x²，再一次 `ct*pt` 完成多项式）
- degree 4：`y = a + b·x + c·x² + d·x³ + e·x⁴`（两次 `ct*ct` 算 x²/x³/x⁴，
  再一次 `ct*pt` 完成多项式）

### 5.1 公共前段：LN 尾部 + Wffn1（所有 degree 共用）

| # | 算子 | 输入 1 | 输入 2 | 输出 |
|---|---|---|---|---|
| 1 | **`ct*ct` = ewmulcc**（normalize） | `x − mean` (ct) | `1/std` (ct) | (x−mean)/std |
| 2 | **`ct*pt` = ewmulcp**（× γ） | normalized | `gama1 : 20` (pt) | γ·(x−mean)/std |
| 3 | Bs Rotation | | | |
| 4 | **`ct*pt` = matmulcp**（**Wffn1**） | normalized | `Wffn1 : 22` (pt) | ffn1_out (ct) |
| 5 | Gs Rotation → `ct : ffn1_out` | | | |

### 5.2 各 degree 的 GeLU 多项式段

#### gelu1（degree = 1，**N = 8192**）

| # | 算子 | 输入 1 | 输入 2 | 输出 |
|---|---|---|---|---|
| 6 | **`ct*pt` = ewmulcp** 完成 `a + b·x` | ffn1_out | `a/b : 20` | gelu_out |

#### gelu2（degree = 2，**N = 16384**）

| # | 算子 | 输入 1 | 输入 2 | 输出 |
|---|---|---|---|---|
| 6 | **`ct*ct` = ewmulcc** 算 `x²` | ffn1_out | ffn1_out (self-mul) | gelu-x² |
| 7 | **`ct*pt` = ewmulcp** 完成 `a + b·x + c·x²` | x, x² 组合 | `a/b/c : 20` | gelu_out |

#### gelu4（degree = 4，**N = 16384**）

| # | 算子 | 输入 1 | 输入 2 | 输出 |
|---|---|---|---|---|
| 6 | **`ct*ct` = ewmulcc** 算 `x²` | ffn1_out | ffn1_out | gelu-x² |
| 7 | **`ct*ct` = ewmulcc** 算 `x³` 和 `x⁴` | gelu-x²、x | self-mul / cross | gelu-x³/x⁴ |
| 8 | **`ct*pt` = ewmulcp** 完成 `a + b·x + c·x² + d·x³ + e·x⁴` | 各次幂的组合 | `a/b/c/d/e : 20` | gelu_out |

> 注：用户图中"`gelu-x⁴/gelu-x³`"用一个 ct*ct 节点同时表示了 `x⁴ = x²·x²` 和
> `x³ = x²·x`（两路并行各算一次乘法）。代码层面 `polynomial(...)` 是用
> `[x.pow(i) for i in range(degree+1)]` 一次性生成所有幂，对应
> 的 fused CKKS 协议把这些步骤合并。

> 项目代码里 `GELU_COEEF` 同时给了 degree 0、1、2、3、4 五套系数。**当前用户允许的
> degree 是 {1, 2, 4}**（参考用户说明）。degree 0 / degree 3 即使代码里有，也不在
> 现行选择集合内；如未来开放，再补 gelu0 / gelu3 子表。

### 5.3 项目里对应的注册表 id（与 degree 无关，按算子聚合）

| BLB 步骤 | 项目 id | op_type | 已加噪？ |
|---|---|---|---|
| 步骤 1 (× 1/std) | `ffn.layernorm.stat_div`（聚合） | stat_div | ❌ |
| 步骤 2 (× γ) | `ffn.layernorm.scale_mul` | elementwise_mul | ❌ |
| 步骤 4 (Wffn1) | `ffn.intermediate_proj` | linear_mm | ✅ `wffn1` |
| 步骤 6/7 (各次幂 x², x³, x⁴) | `ffn.gelu.power` | self_power | ❌ |
| 步骤 6/7/8 (各 coeff_i × x^i) | `ffn.gelu.coeff_mul` | scalar_mul | ❌ |

> 同样地，`ffn.gelu.power` / `ffn.gelu.coeff_mul` 当前是聚合 id，**不区分 degree**。
> 若你需要 "只对 gelu4 的 x³ 步加噪声" 这种粒度，告诉我，我会拆成
> `ffn.gelu.power.d{1,2,4}.s{...}` 之类的细粒度 id。

### 5.4 Block 5 总噪声候选清单（按 degree）

```
所有 degree 共有
[未加噪] ffn.layernorm.stat_div     → (x-mean) / (1/std) 操作数
[未加噪] ffn.layernorm.scale_mul    → normalized / γ 操作数
[已加噪] ffn.intermediate_proj      → W_ffn1 操作数  (X 操作数未加)

按 degree 区分 GeLU 多项式段
[未加噪] ffn.gelu.power · d=1   →  无 ct*ct（线性多项式）
[未加噪] ffn.gelu.power · d=2   →  step 1: x²
[未加噪] ffn.gelu.power · d=4   →  step 1: x²,  step 2: x³ 与 x⁴ (并行)

[未加噪] ffn.gelu.coeff_mul · d=1   →  最终 ct*pt: a/b
[未加噪] ffn.gelu.coeff_mul · d=2   →  最终 ct*pt: a/b/c
[未加噪] ffn.gelu.coeff_mul · d=4   →  最终 ct*pt: a/b/c/d/e
```

---

## 6. Block 之外：Embeddings / Pooler / Classifier（一次性）

不在 BLB Figure 10 的 5-block 循环内，但也是乘法操作的候选。

### 6.1 Stage 0：BertEmbeddings（layer 之前一次）

| 项目 id | op_type | BLB 算子类 | 已加噪？ |
|---|---|---|---|
| `emb.word_lookup` | embedding_lookup | matmulcp（OneHot · W） | ❌ |
| `emb.token_type_lookup` | embedding_lookup | matmulcp | ❌ |
| `emb.position_lookup` | embedding_lookup | matmulcp | ❌ |
| `emb.layernorm.stat_div` | stat_div | sum + smulcp + ewmulcc 等聚合 | ❌ |
| `emb.layernorm.scale_mul` | elementwise_mul | smulcp（× γ） | ❌ |

### 6.2 Stage 3 / 4：Pooler + Classifier（最后一次）

| 项目 id | op_type | BLB 算子类 | 已加噪？ |
|---|---|---|---|
| `pooler.dense` | linear_mm | matmulcp（[CLS]·W_pool） | ❌ |
| `head.classifier` | linear_mm | matmulcp（pooled·W_clf） | ❌ |

---

## 7. 全部 id 速查（按 BLB block 排列）

```
─── Embeddings (一次性) ─────────────────────────────────
emb.word_lookup
emb.token_type_lookup
emb.position_lookup
emb.layernorm.stat_div
emb.layernorm.scale_mul

─── Per-layer 5-block 循环 ────────────────────────────
═ Block 1 (Wffn2 + LN head) ═
ffn.output_proj                  [✅ wffn2]
ffn.layernorm.stat_div           [所属：上一层 Block 1 头部 / 当前层 Block 4 / 当前层 Block 1]

═ Block 2 (LN tail + Wq/Wk/Wv + Q·K^T) ═
attn.layernorm.stat_div          (× 1/std 部分)
attn.layernorm.scale_mul         (× γ 部分)
attn.q_proj                      [✅ wq]
attn.k_proj                      [✅ wk]
attn.v_proj                      [✅ wv]   (🚧 实际 fuse 位置 TBD)
attn.qk_matmul
attn.qk_scale_div                (我们项目在 softmax 前显式做 / sqrt(Dh))

═ Block 3 (Softmax exp 近似主体, degree 决定步数) ═
attn.softmax.expapprox.scalar_div          (1 + x/2^n 那一步 ct*pt)
attn.softmax.expapprox.power · d=2..6      (degree 次 ct*ct 自乘)
attn.softmax.norm_div                       (exp_out / sum_exp，含 rec 非线性)
attn.head_mask_mul                          (默认不触发)

═ Block 4 (softmax·V + Wo + LN head) ═
attn.probs_v_matmul              [✅ softmax_probs / value_after_softmax]
attn.o_proj                       [✅ wo]
attn.layernorm.stat_div           (本层 Block 4 头部，与 Block 1 共享同一个 id)

═ Block 5 (LN tail + Wffn1 + GELU, degree 决定步数) ═
ffn.layernorm.stat_div            (× 1/std 部分)
ffn.layernorm.scale_mul           (× γ 部分)
ffn.intermediate_proj             [✅ wffn1]
ffn.gelu.power · d=1, 2, 4
ffn.gelu.coeff_mul · d=1, 2, 4

─── Pooler / Classifier (一次性) ──────────────────────
pooler.dense
head.classifier
```

---

## 8. 当前已加噪声的 id（**完整列表**）

| 现有噪声键 (`config/constants.py:NOISE_KEYS` + softmax/V 钩子) | 注入位置（`function_handler.py`） | 影响的 id（按 BLB block） |
|---|---|---|
| `'x'` | `BertLayer.forward` 入口包装 | Block 2 的 `attn.q_proj` / `attn.k_proj` / `attn.v_proj` 的 X 操作数 |
| `'wq'` | `attention.self.query.forward` | Block 2 `attn.q_proj` 的 W 操作数 |
| `'wk'` | `attention.self.key.forward` | Block 2 `attn.k_proj` 的 W 操作数 |
| `'wv'` | `attention.self.value.forward` | Block 2 `attn.v_proj` 的 W 操作数 |
| `'wo'` | `attention.output.dense.forward` | Block 4 `attn.o_proj` 的 W 操作数 |
| `'wffn1'` | `intermediate.dense.forward` | Block 5 `ffn.intermediate_proj` 的 W 操作数 |
| `'wffn2'` | `output.dense.forward` | Block 1 `ffn.output_proj` 的 W 操作数 |
| `'softmax_probs'` | `_apply_softmax_value_noise` | Block 4 `attn.probs_v_matmul` 的 probs 操作数 |
| `'value_after_softmax'` | `_apply_softmax_value_noise` | Block 4 `attn.probs_v_matmul` 的 V 操作数 |

scaling factor 取值集合：
```
INPUT_NOISE_ALLOWED_SCALING_FACTORS  = (22, 24, 26, 28, 30)        # 'x'
WEIGHT_NOISE_ALLOWED_SCALING_FACTORS = (14, 16, 18, 20, 22)        # 'wq','wk','wv','wo','wffn2'
WFFN1_NOISE_ALLOWED_SCALING_FACTORS  = (16, 18, 20, 22, 24)        # 'wffn1'
SOFTMAX_VALUE_NOISE_ALLOWED_SCALING_FACTORS = (10..48 偶数)         # softmax_probs / value_after_softmax
```

---

## 9. 还没加噪声的 id（按 BLB block 分组，方便你点名）

> 报噪声指令格式建议：
> ```
> 在 id = <id 名>，对操作数 = <X / W / γ / 1/D / 1/(2^n) / coeff / ...> 加噪声，
> scaling factor = <K>，分布 = <encoding/fresh/rescale>
> [条件：仅当 softmax degree = D / GELU degree = D 时]
> ```

### Block 1（Wffn2 + LN head）
- `ffn.output_proj` 的 **X (= GELU_out) 操作数**
- `ffn.layernorm.stat_div` 头部三段：mean 步、(x−mean)² 步、variance 步

### Block 2（LN tail + Wq/Wk/Wv + Q·K^T）
- `attn.layernorm.stat_div` 的 (x−mean) / (1/std) 操作数
- `attn.layernorm.scale_mul` 的 normalized / γ 操作数
- `attn.q_proj` / `attn.k_proj` / `attn.v_proj` 的 **X 操作数**（W 已加噪）
- `attn.qk_matmul` 的 **Q 操作数 / K^T 操作数**
- `attn.qk_scale_div` 的 **attention_scores / (1/√Dh) 操作数**

### Block 3（Softmax）—— 与 degree 关联
- `attn.softmax.expapprox.scalar_div`：x_shifted、1/(2^n) 操作数（所有 degree 都有）
- `attn.softmax.expapprox.power`：每次 `ct*ct` 的两个 self-mul 操作数
  - degree=2：第 1、2 次平方
  - degree=3：第 1、2、3 次平方
  - degree=4：第 1、2、3、4 次平方
  - degree=5：第 1、2、3、4、5 次平方
  - degree=6：第 1、2、3、4、5、6 次平方
- `attn.softmax.norm_div` 的 exp_out 操作数

### Block 4（softmax · V + Wo + LN head）
- `attn.probs_v_matmul`：当前已加噪，但若想做 *单独的 scaling factor 调优* 仍然可点名
- `attn.o_proj` 的 **X (= softmax·V) 操作数**（W_o 已加噪）
- `attn.layernorm.stat_div` 头部三段（同 Block 1 头部）

### Block 5（LN tail + Wffn1 + GELU）—— 与 degree 关联
- `ffn.layernorm.stat_div` 的 (x−mean) / (1/std) 操作数
- `ffn.layernorm.scale_mul` 的 normalized / γ 操作数
- `ffn.intermediate_proj` 的 **X 操作数**（W_ffn1 已加噪）
- `ffn.gelu.power`：每次 `ct*ct` 的两个操作数
  - degree=1：无（线性）
  - degree=2：x² 那一次
  - degree=4：x² 一次、x³/x⁴ 一次（共 2 次）
- `ffn.gelu.coeff_mul`：最终 `ct*pt` 的 powers / coeff 操作数
  - degree=1：a/b 系数共 2 个
  - degree=2：a/b/c 系数共 3 个
  - degree=4：a/b/c/d/e 系数共 5 个

### Embeddings / Pooler / Head
- `emb.*` 全部
- `pooler.dense`、`head.classifier` 的 X 与 W 操作数

---

## 10. 后续协作约定

1. **新增噪声候选**：先改 [`noise_targets_registry.py`](../noise_targets_registry.py) 的
   `NOISE_TARGETS` 列表（schema 见文件顶部 docstring），再在本 md 第 1–6 节相应 block
   的表里追加一行；
2. **细粒度子 id**：当前 Block 3、Block 5 的 softmax / GELU 是聚合 id。如果你后面要
   按 degree × step 加噪声，告诉我，我会一次性把它们拆成形如
   `attn.softmax.expapprox.power.d6.s4` / `ffn.gelu.power.d4.s2` 的子 id；
3. **报噪声指令**：用第 9 节顶部的格式描述要加在哪个 id 的哪个操作数；
4. **TBD 区域**：第 1.2 / 2.1 / 4.1 节里标 🚧 TBD 的位置，等你补完用户图后告诉我，
   我会更新本 md 与注册表；
5. **N=8192 / N=16384 噪声表**：当前 `function_handler.py` 只有一张
   `INPUT_NOISE_VARIANCE_TABLE`。如果 BLB 不同 block 用不同 N 的噪声表，
   等你确认后我会把它扩成 `INPUT_NOISE_VARIANCE_TABLE_N{8192,16384}`，并在 md 第 0.3 节
   表格里给每个 block 标精确的"用哪张表"。
