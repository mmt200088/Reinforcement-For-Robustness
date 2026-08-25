# Rescale Optimizer — 7 算法总览

## 符号表

| 符号 | 说明 |
|------|------|
| $N$ | 多项式模数度数 |
| $Q$ | RNS-CKKS 的系数模数链 |
| $q_i$ | 模数链中的第 $i$ 个素模 |
| $q_{\text{head}}$ | 模数链的头素模（计入 ActiveBits） |
| $q_{\text{tail}}$ | 特殊尾素模，用于 key-switching / rotation（**不**计入 ActiveBits） |
| $\mathcal{Q}_{\text{legal}}$ | 合法素模 bit 宽度集合，SEAL 中 $\mathcal{Q}_{\text{legal}}=\{q\in\mathbb{Z}:30\le q\le 60\}$ |
| $q_{\max}$ | $\mathcal{Q}_{\text{legal}}$ 中最大合法 bit 宽度（SEAL 下 $q_{\max}=60$） |
| $L$ | 模数链可用 level 数 |
| $\Delta$ | CKKS 明文 scaling factor，每个数据（pt/ct 边）都自带一个 |
| $h_{\text{sf}}$ | 均匀 scale headroom，加在每个 $t_j^{\text{base}}$ 之上 |
| $A_j^{\text{budget}}$ | cut point $c_j$ 的幅度预算（bit） |
| $c_0, c_1, \ldots, c_M, c_{M+1}$ | 按拓扑序的 cut points；$c_{M+1}$ 是 dummy sink |
| $\text{Amp}_j$ | 在 $c_j$ 的幅度分布 $(\mathbf{p}_j, \mathbf{a}_j)$ |
| $\text{SNR}_j$ | 在 $c_j$ 的 SNR 要求 $(p_j, \varepsilon_j)$ |
| $\mathcal{N}[\text{op}, s]$ | 噪声查表，给定操作和 sf_bits 返回噪声上界 |

---

## 计算图节点分类

| 类型 | 可 rescale | 计入成本 | 是否 cut point |
|------|:-:|:-:|:-:|
| `SOURCE`     (= $c_0$，初始操作数 / fresh ct 或 pt-like) | ✗ | — | ✓ (index 0) |
| `CTCT_MUL`   ($ct \times ct$) | ✓ | ✓ | ✓ |
| `CTPT_MUL`   ($ct \times pt$) | ✓ | ✓ | ✓ |
| `ROTATION`   | — | ✓ | ✗ |
| `PT_OP`      | — | ✓ | ✗ |
| `PT`（明文叶子操作数） | — | — | ✗ |
| `DUMMY_SINK` (= $c_{M+1}$，虚拟终点) | — | — | 虚拟（索引 $M+1$） |

**重要语义**
- $c_0$（SOURCE）是**初始操作数**，携带一个 baseline scaling factor $t_0$，但本身**不能 rescale**——第一次 rescale 只能在第一个乘法 $c_1$ 处发生。
- $c_M$ 就是**最后一个乘法节点**（CTCT_MUL 或 CTPT_MUL），没有独立的 "SINK" 类型。
- $c_{M+1}$ 是虚拟 dummy sink，用于表达 "在最后一个 rescale 后还能再做若干乘法但不超过 amplitude budget" 这种不做最后一次 rescale 的情形（即 tail 边）。
- Skeleton $S^\star = [s_0=0, s_1, \ldots, s_R, M+1]$ 共有 $R$ 次 rescale，分别发生在 $s_1, \ldots, s_R$（全是乘法节点）；$s_0 = 0$ 只是起点状态。

每个节点都带一个 `scale_delta_bits`：

- `SOURCE` / `PT` / `ROTATION` / `PT_OP` / `DUMMY_SINK`: $0$
- `CTPT_MUL`: $\Delta_{pt}$（pt 操作数的 scaling factor）
- `CTCT_MUL`: **忽略**。真实 delta 由 `PropagateScale` 按下述规则动态计算。

此外 `CTCT_MUL` 节点有一个可选字段 `other_ct_scale_bits`：

- `None`（缺省）：对称 squaring 语义，两个 ct 操作数都在当前工作 scale 上 → $s \to 2 \cdot s$。
- 非 `None`：非对称 CTCT，外部 ct 以该固定 scale 进来 → $s \to s + \text{other\_ct\_scale\_bits}$。

## PropagateScale

$$
s_{\text{pre}}(i,j) = \text{PropagateScale}(t_i, \text{nodes}_{i \to j})
$$

```
s ← t_i
for node in path(c_i, c_j]:
    if node.type == CTCT_MUL:
        if node.other_ct_scale_bits is not None:
            s ← s + node.other_ct_scale_bits      # 非对称 CTCT
        else:
            s ← 2 · s                             # 对称 squaring
    else:
        s ← s + node.scale_delta_bits             # CTPT_MUL / non-mul nodes
return s
```

- 路径 $\text{path}(c_i, c_j]$ 不含起点 $c_i$：
  - 若 $i \ge 1$，$t_i$ 已经是 $c_i$ 做完 rescale 之后的 scale；
  - 若 $i = 0$（SOURCE），$t_0$ 直接是初始操作数的 baseline scaling factor（c_0 没有 rescale）。
- 包含终点 $c_j$（乘法节点自己会为 scale 贡献 delta），所以 $s_{\text{pre}}$ 是乘法后、rescale 前的 scale。

> **注**（CTCT 与 $q_{\max}$）：自平方语义下 $s \to 2s$，当 $t_i$ 已经接近 $q_{\max}$ 时
> 可能 $s_{\text{pre}}(i, j) - t_j > q_{\max}$ ⇒ 该 stage 边在 Feasibility-DAG
> 中不可行。通常靠更早的 rescale 将 $t$ 压低即可，否则就要调大 $h_{\text{sf}}$ /
> 放宽 $\mathcal{Q}_{\text{legal}}$。非对称 CTCT（`other_ct_scale_bits` 被设置）
> 相当于把外部 ct 的 scale 当作一个固定的 "pt-like $\Delta$"，行为更像
> CTPT_MUL，但消耗一个 ct × ct 乘法 level。

---

## Algorithm 1 — Feasibility-DAG Construction

**Input**
- 计算图 $G$，cut points $c_0, \ldots, c_M$，dummy sink $c_{M+1}$
- $\text{Amp}_j = (\mathbf{p}_j, \mathbf{a}_j)$，$\text{SNR}_j = (p_j, \varepsilon_j)$
- 噪声表 $\mathcal{N}[\text{op}, s]$
- 均匀 headroom $h_{\text{sf}}$，amplitude budgets $\{A_j^{\text{budget}}\}$

**Output**
- Feasibility DAG $\mathcal{G}_{\text{feas}} = (V, \mathcal{E}_{\text{feas}})$

```
V ← {0, 1, ..., M, M+1}
E_feas ← ∅

# Step 1: baseline scale for every real cut point
for j = 0 to M:
    (p_j, ε_j) ← SNR_j
    a_j       ← Amp_j.interpolate(p_j)
    t_j       ← FindMinSF(N, op_j, a_j, ε_j) + h_sf

# Step 2a: stage edges (i, j),  j ≤ M
for i = 0 to M-1:
    for j = i+1 to M:
        nodes ← 路径 c_i → c_j 上所有节点
        s_pre(i,j) ← PropagateScale(t_i, nodes)
        d(i,j)     ← s_pre(i,j) - t_j
        if d(i,j) ∈ Q_legal:
            E_feas ← E_feas ∪ {(i,j)}

# Step 2b: tail edges (i, M+1)
for i = 0 to M:
    γ_tail(i) ← max_{i < v ≤ M} { PropagateScale(t_i, nodes_{i→v}) + A_v^budget }
    if γ_tail(i) < q_max:
        E_feas ← E_feas ∪ {(i, M+1)}
```

`FindMinSF` 从小到大遍历 $\mathcal{N}[\text{op}]$ 的 sf_bits，返回首个使
$\mathcal{N}[\text{op}, s] / a \le \varepsilon$ 的 $s$（找不到则返回最大值）。

> **注** 伪代码里 `γ_tail(i)` 的求 max 对 $v$ 取值上界取为 $M$（$v = M+1$ 时
> $A^{\text{budget}}_{M+1}$ 未定义，故排除）。当 $i = M$ 时区间为空，约定
> $\gamma^{\text{tail}}(M) = 0$，tail 边平凡可行。

---

## Algorithm 2 — Forward/Backward Reachability

**Input** $\mathcal{G}_{\text{feas}}$

**Output**
- $\text{FwdSteps}[j]$, $\text{BwdSteps}[j]$
- 可行性谓词 $\text{Feas}(i,j,l)$

```
E_stage ← {(i,j) ∈ E_feas : j ≤ M}
E_tail  ← {(i,M+1) ∈ E_feas}

# Forward
FwdSteps[0] ← {0}
for j = 1 to M:
    for each (i,j) ∈ E_stage:
        for each r' ∈ FwdSteps[i]:
            if r'+1 ≤ M:
                FwdSteps[j] ∪= {r'+1}

# tail 边不加 level
for each (i, M+1) ∈ E_tail:
    for each r' ∈ FwdSteps[i]:
        FwdSteps[M+1] ∪= {r'}

# Backward
BwdSteps[M+1] ← {0}
for each (i, M+1) ∈ E_tail:
    BwdSteps[i] ∪= {0}

for j = M down to 0:
    for each (j,k) ∈ E_stage:
        for each q' ∈ BwdSteps[k]:
            if q'+1 ≤ M:
                BwdSteps[j] ∪= {q'+1}

# Feasibility predicate
for each (i,j) ∈ E_stage:
    for l = 1 to M-i:
        Feas(i,j,l) ← (FwdSteps[i] ≠ ∅) ∧ (l-1 ∈ BwdSteps[j])

for each (i, M+1) ∈ E_tail:
    Feas(i, M+1, 0) ← (FwdSteps[i] ≠ ∅)
```

Tail 边只有 $l = 0$ 时可用（terminal level exhaustion）。

---

## Algorithm 3 — Backward Level-DP

**Input** $\mathcal{G}_{\text{feas}}$, $\text{FwdSteps}$, $\text{Feas}$, cost params $(\lambda_0, \lambda_1, \alpha, \beta)$

**Output** 最优 skeleton $S^{\text{best}}$，最优代价 $C^{\text{best}}$，初始 level $L^{\text{best}}$

**统一边成本：**
$$
\widetilde{C}(i,j,l) = \begin{cases}
\lambda_0 + \lambda_1 l + \alpha \cdot E(i,j,l) + \beta \cdot d(i,j), & j \le M \\
\lambda_0 + \alpha \cdot E_{\text{tail}}(i, 0), & j = M+1
\end{cases}
$$

**下一 level 规则：**
$$
\widetilde{\ell}(j, l) = \begin{cases} l - 1, & j \le M \\ 0, & j = M+1 \end{cases}
$$

（tail 边不消耗 level；但 tail 边仅在 $l = 0$ 时可用 → 强制所有 level 用完后才能走 tail。）

```
DP[M+1, 0] ← 0, DP[i, l] ← +∞ otherwise
for l = 0 to M:
    for i = M down to 0:
        Cand(i, l) ← { j : Feas(i, j, l) }
        if Cand(i, l) ≠ ∅:
            DP[i, l] ← min over j ∈ Cand(i, l) of
                       C̃(i,j,l) + DP[j, ℓ̃(j, l)]
            Next[i, l] ← argmin

L* ← argmin over L ∈ FwdSteps[M+1] of DP[0, L]
S* ← BacktrackBackward(Next, 0, L*)
```

---

## Algorithm 4 — ConstructModulusChain (top-level)

**Input** 初始 skeleton $S^\star$，baseline scales $\mathbf{t}^{\text{base}}$，初始素模 bit $\hat{\mathbf{d}}$，$\text{Feas}$，$\text{FwdSteps}[M+1]$，cost params，$h_{\text{sf}}$，$\{A_j^{\text{budget}}\}$

**Output** $S^{\text{final}}$，$\mathbf{q}^{\text{final}}$，$\mathbf{t}^{\text{final}}$，validity flag

```
# Step 1: Initialize
for r = 0 to R:  t_r ← t_r^base + h_sf
for r = 1 to R:  bits(q_r) ← d̂_r
q ← [q_head, q_1, ..., q_R, q_tail]

# Step 2: Try to repair
(q, t, flag) ← RepairChain(S*, q, t, t^base, {A_j^budget}, Q_legal)
if flag = Valid:
    S_cand, q_cand, t_cand, t_base_cand ← S*, q, t, t^base
else:
    # Step 3: K-best fallback
    (S_cand, t_base_cand, q_cand, t_cand, flag) ←
        BestFirstRepairableSkeleton(...)
    if flag = Invalid:
        return Nil, Nil, Nil, Invalid

# Step 4: Compress unused headroom
(t_final, q_final) ← CompressHeadroom(G, S_cand, t_cand, t_base_cand, q_cand, Q_legal)

# Step 5: Final consistency check
(valid, _, _, _) ← ValidateCutPoints(S_cand, q_final, t_final, {A_j^budget})
if not valid:
    return Nil, q_final, t_final, Invalid

return S_cand, q_final, t_final, Valid
```

---

## Algorithm 5 — RepairChain  (chain-consistent variant)

**Input** skeleton $S^\star$，chain $\mathbf{q}$，scales $\mathbf{t}$，baseline $\mathbf{t}^{\text{base}}$，$\{A_j^{\text{budget}}\}$

**Output** 修复后 chain $\mathbf{q}$，scales $\mathbf{t}$，validity flag

```
while true:
    (valid, j*, r*, Δ*) ← ValidateCutPoints(S*, q, t, {A_j^budget})
    if valid:   return q, t, Valid

    repaired ← False
    for u = r*+1 to R:
        q_u' ← next larger legal prime of q_u in Q_legal
        if q_u' = ⊥:  continue

        # Try the bump and re-walk t[u..R] chain-consistently
        q̃ ← q with q̃_u = q_u'
        t̃[0..u-1] ← t[0..u-1]
        for v = u to R:
            sf_pre_v ← PropagateScale(t̃[v-1], path(s_{v-1} → s_v))
            t̃[v]    ← sf_pre_v - bits(q̃_v)
            if t̃[v] < t_v^base:    # cannot absorb
                ok ← False; break
        if ok:
            q ← q̃; t ← t̃; repaired ← True; break

    if not repaired:
        return q, t, Invalid
```

**几何理解**：把 $q_u$ 变大相当于让中间 level 留出更多 headroom。代价是
$t_u$ 跌 $\delta$，但**对 $v > u$，$t_v$ 跌 $2^{k_v} \cdot \delta$**
（$k_v$ = 从 stage $v-1$ 到 $v$ 的路径上对称 $\textsf{CTCT\_MUL}$ 数量）。
旧版用 "$t_v \mathrel{-}= \delta$" 的线性公式漏掉了 CTCT 的倍增，会
让 $t_v$ 看起来够 baseline，实际部署到 SEAL 之后的 ct scale 反而**低于
baseline**。本版每次 bump 都用 `PropagateScale` 重走一遍 $t_u, \ldots,
t_R$，从而保持

$$t_v \;=\; \mathrm{PropagateScale}(t_{v-1}, \text{path}_v) \;-\; \mathrm{bits}(q_v)$$

这一不变量。若任何 $v$ 不能再压就跳过该 $u$；若所有 $u > r^*$ 都不能修，则 Invalid。

---

## Algorithm 6 — BestFirstRepairableSkeleton

**思想** 当初始最优 skeleton 无法 repair，用**偏离约束子问题**搜索次优。

把 $S^\star = (e_1, \ldots, e_T)$ 视为决策序列。对每个 $t$：
- 固定前缀 $(e_1, \ldots, e_{t-1})$
- 禁用第 $t$ 条边 $e_t$
- 在 "偏离约束" $\text{Feas}^{(t)}$ 下重新跑 Backward DP，得到 $(S', C', L')$
- 把 $(C', S', L')$ 压入 min-heap $\mathcal{H}$

```
H ← ∅
U ← {S*}
enqueue deviations of S*

while H ≠ ∅:
    pop (C, S, L)
    derive t^base(S), d̂(S) from S
    t, q ← Initialize using h_sf and d̂
    (q, t, flag) ← RepairChain(S, q, t, t^base(S), {A_j^budget}, Q_legal)
    if flag = Valid:
        return S, t^base(S), q, t, Valid
    # 失败则继续扩展
    enqueue deviations of S into H (skip visited)

return Nil, Nil, Nil, Nil, Invalid
```

min-heap 按 DP 代价排序，保证**第一次**弹出可 repair 的候选就是次优（之后同理）。

---

## Algorithm 7 — CompressHeadroom  (chain-consistent variant)

**Input** graph $G$, skeleton $S^\star$, scales $\mathbf{t}$, baseline $\mathbf{t}^{\text{base}}$, chain $\mathbf{q}$

**Output** 压缩后 $\mathbf{t}'$，$\mathbf{q}'$，**chain-consistent**：
$t'_r = \mathrm{PropagateScale}(t'_{r-1}, \text{path}_r) - \mathrm{bits}(q'_r)$

```
t'_0 ← t_0^base                         # 完全压扁 source 头部空间

for r = 1 to R:
    sf_pre_r ← PropagateScale(t'_{r-1}, path(s_{r-1} → s_r))
    q_cand   ← sf_pre_r - t_r^base       # 最大压缩对应的 prime 位宽
    q_lower  ← q_legal_min
    q_upper  ← min(q_legal_max, bits(q_r))

    if q_lower ≤ q_cand ≤ q_upper:       # 最大压缩可行
        bits(q'_r) ← q_cand
        t'_r       ← t_r^base
    elif q_cand > q_upper:                # 单 stage 砍不动，把头剩在 t' 里
        bits(q'_r) ← q_upper
        t'_r       ← sf_pre_r - q_upper
    else:                                  # q_cand < q_legal_min, 该 stage 不能压
        bits(q'_r) ← bits(q_r)
        t'_r       ← max(t_r, sf_pre_r - bits(q_r))
```

**不变量**

- 链一致性：$t'_r = \mathrm{PropagateScale}(t'_{r-1}, \text{path}_r) - \mathrm{bits}(q'_r)$
- $\mathrm{bits}(q'_r) \in [q_{\text{legal,min}}, \mathrm{bits}(q_r)]$ 且 $\le q_{\text{legal,max}}$
- $t'_r \ge t_r^{\text{base}}$ —— SNR 仍 OK

**与旧线性公式的差异**

旧公式 $d'_r = \mathrm{bits}(q_r) - (c_{r-1} - c_r)$ 隐含
$\mathrm{PropagateScale}(t-\delta, \text{path}) = \mathrm{PropagateScale}(t, \text{path}) - \delta$，
但路径上每出现一个对称 $\textsf{CTCT\_MUL}$（$s \to 2s$）都会让减量变成
$2\delta$。$k$ 个对称 CTCT 时，旧版会留下 $(2^k - 1) \cdot c_{r-1}$ 比特的
chain slack，且 $t'_r$ 与真正部署到 SEAL 后的 post-rescale scale 不一致。
新版用 `PropagateScale` 算真实 $sf_{\text{pre}}$，从 $t'_{r-1}$ 链式推出
$t'_r$，从而**输出可直接喂给 SEAL/OpenFHE**。

> 最终 `ConstructModulusChain` 的 Step 5 会再跑一遍 `ValidateCutPoints`，
> 保证 compression 没有因为浮点边界破坏合法性。

---

## Algorithm 8 — ValidateCutPoints

**Input** skeleton $S^\star = [s_0, s_1, \ldots, s_R, M+1]$，chain $\mathbf{q}$，scales $\mathbf{t}$，$\{A_j^{\text{budget}}\}$

**Output** validity flag，首个违规 $j^\star$，对应 stage $r^\star$，缺口 $\Delta^\star$

```
for j = 0 to M:
    r      ← max{ u ∈ 0..R : s_u ≤ j }
    ŝ_j    ← PropagateScale(t_r, nodes_{s_r → j})
    B_j^act ← ActiveBits(S*, q, j)
    Δ_j    ← ŝ_j + A_j^budget - B_j^act
    if Δ_j > 0:
        return False, j, r, Δ_j
return True, Nil, Nil, 0
```

**ActiveBits 定义**
$$
B_j^{\text{act}} = \text{bits}(q_{\text{head}}) + \sum_{u = r+1}^{R} \text{bits}(q_u)
$$
（$q_{\text{tail}}$ 不计入，因为它只用于 key-switching / rotation。）

**直觉**：到 cut point $c_j$ 为止，共已完成 $r$ 次 rescale（即 $q_1, \ldots, q_r$ 被消耗），剩余 modulus = $q_{\text{head}} + q_{r+1} + \cdots + q_R$。此剩余必须容得下当前 scale $\hat{s}_j$ 与 amplitude budget $A_j^{\text{budget}}$，否则 overflow。

---

## 整体流水线

```
 ┌─────────────────────────────────────┐
 │ Alg 1  Feasibility DAG              │
 │ Alg 2  Reachability                 │
 │ Alg 3  Backward Level-DP  → (S*, L*)│
 │ Alg 4  ConstructModulusChain        │
 │   ├─ Alg 5  RepairChain (on S*)     │
 │   │    ├─ Alg 8  ValidateCutPoints  │
 │   ├─ Alg 6  BestFirstRepairableSkel │
 │   │         ↳ Alg 5 (fallback)      │
 │   ├─ Alg 7  CompressHeadroom        │
 │   └─ Alg 8  final check             │
 └─────────────────────────────────────┘
```

**失败判定**：
- Alg 1 后 stage_edges 与 tail_edges 均为空 → infeasible；
- Alg 2 后 $\text{FwdSteps}[M+1] = \emptyset$ → infeasible；
- Alg 4 returns `Invalid` when $h_{\text{sf}}$, $A_j^{\text{budget}}$, or $\mathcal{Q}_{\text{legal}}$ violates the chain constraints.
