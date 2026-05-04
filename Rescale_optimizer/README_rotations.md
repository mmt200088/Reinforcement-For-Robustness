# Rotation 节点 & “Effective Rotation” 说明

这份文档专门解释 ROTATION 节点在 graph / 优化器 / 输出文件里的角色，
重点是 `static_skeletons_*.json` / `replan_*.json` 中新增字段
`effective_rotations` 的定义与判定流程。和 `README_configs.md` 里的 ②/④
schema 配套阅读。

---

## 1. ROTATION 节点是什么

`NodeType.ROTATION` 表示一次密文旋转（galois automorphism），是 HE 中的
非乘法操作但 **有显著的 keyswitching 计算开销和 KS 噪声**。在算法层面：

- `is_cut_point = False`，**不会** 单独占一个 cut point；只能挂在某个
  stage 的 `nodes` 列表里。
- `is_rescalable = False`，**不能** 在 rotation 后插 rescale。
- `scale_delta_bits = 0`，propagate_scale 经过它时 scale 不变。
- `count` ≥ 1：表示同一位置上重复了几次 rotation（比如多次 partial-sum
  rotation 折叠到一个节点）。
- `cost_slope * level + cost_intercept`：每次 rotation 的成本与执行时
  剩余的模数 level 成正比。

---

## 2. 在 config 文件里怎么写

ROTATION 必须挂在某个 stage 的 `nodes[]` 数组里（不能作为 cut_point）：

```json
{
  "_comment": "stage 2 -> c_3: ct*pt with Wffn1 plaintext (sf = 22) -> ffn1_out.",
  "nodes": [
    {
      "name": "rot_bs_wffn1",
      "type": "ROTATION",
      "count": 1,
      "scale_delta_bits": 0,
      "cost_slope": 0.5,
      "cost_intercept": 6.0
    }
  ],
  "cut_point": {
    "name": "ctpt_wffn1",
    "type": "CTPT_MUL",
    "scale_delta_bits": 22,
    "count": 1,
    "cost_slope": 0.5,
    "cost_intercept": 8.0
  }
}
```

含义：上面的 rotation `rot_bs_wffn1` 落在 stage 2，即拓扑序上位于
`c_2` 与 `c_3` 之间，**先于 `ctpt_wffn1` 这个乘法 cut point 执行**。

---

## 3. node_id / stage_anchor 的语义

graph 装载（`config_loader.py`）时按拓扑序逐个分配 `node_id`，顺序是：

```
SOURCE → stage₀.nodes[*] → stage₀.cut_point  (= c_1)
       → stage₁.nodes[*] → stage₁.cut_point  (= c_2)
       → ...
       → stage_{M-1}.nodes[*] → stage_{M-1}.cut_point (= c_M)
       → DUMMY_SINK (虚拟 c_{M+1})
```

- `node_id` = 在这个序列里的下标（也等于 `topo_order`）。
- `stage_anchor`：
  - 对 ROTATION / 普通节点：等于它所在 stage 的下标 `k`，表示节点位于
    `(c_k, c_{k+1}]`。
  - 对 cut_point 节点：等于该 cut point 自身的索引（c_k 的 `stage_anchor = k`）。

这样可以很容易地反查每个 ROTATION 上游紧邻的 cut point — 就是 `c_k`。

---

## 4. “有效 rotation” 的形式化定义

设当前 skeleton 为

```
skeleton = [c_0=SOURCE, s_1, s_2, ..., s_R, c_{M+1}=DUMMY_SINK]
```

其中 `s_1, ..., s_R` 是 R 次 rescale 的 cut point 索引。
**rescale cut point 集合 = `skeleton[1:-1] = {s_1, ..., s_R}`**。

定义：

> **一个 ROTATION 节点 N 是 effective ⇔ 它的 `stage_anchor = k` 且 `k ∈ skeleton[1:-1]`**（也就是 `c_k` 是某次 rescale 的 cut point）。

**为什么这个判据等价于“紧跟 rescale，中间没有夹其它非 rescale cut point”：**

- 拓扑序上 cut point **只在** stage 边界出现，N 与 c_k 之间不可能再出现
  任何其它 cut point。中间能出现的只是 stage k 早于 N 的同类节点
  （rotation / pt 叶 / pt_op），它们都不改 scale 也不是 cut point。
- 所以 N 与上游最近的 cut point 一定就是 `c_k`。
- 当 `c_k` 是 rescale cut point（`k ∈ skeleton[1:-1]`），N 在 c_k 的
  rescale 操作之后立即执行。
- 当 `c_k` 是 SOURCE（`k = 0`）或非 rescale 乘法 cut point
  （`1 ≤ k ≤ M`、但 `k ∉ skeleton`）时，N 与上游最近的 rescale 之间
  必然夹着至少一个 “非 rescale 的乘法 cut point”，故 N 是 ineffective。

---

## 5. 有效 rotation 的 sf 怎么算

如果 N 是 effective，且 `c_k` 在 skeleton 里是第 `r` 次 rescale
（即存在 `r ∈ [1, R]` 使得 `s_r = k`），那么：

- c_k 的 rescale 完成后工作 sf 立刻降到 `cut_point_sf[k].sf_post`
  （即文档 `README_configs.md` 中的 “chain-consistent t_r” / `cr.t[r]`）。
- ROTATION 的 `scale_delta_bits = 0`，它在 c_k 的 rescale 后立即跑，
  scale 不变。
- 所以 **N 的执行时 sf = `cut_point_sf[k].sf_post`**。

如果 N 是 ineffective（`k ∉ skeleton[1:-1]`）：

- 它的执行 sf = `cut_point_sf[k].sf` —— 即上一个 rescale 之后，
  途经一系列非 rescale 乘法 cut point 累加得到的高 scale。
- 在 mrpc 实例里，这个 sf 通常在 50–88 bit；而 effective rotation
  的 sf 一般在 30–34 bit。

---

## 6. 输出字段 `effective_rotations`

`scripts/batch_run_configs.py` 的 ② 和 `scripts/replan_what_if.py` 的 ④
（`new_compact_config` 块）都会输出：

```json
"effective_rotations": [
  {"node_id": 5, "name": "rot_gs_wffn1", "after_cut_point": 3, "sf": 31, "count": 1}
]
```

| key                | 含义 |
| ------------------ | --- |
| `node_id`          | graph 全局 id（与 `propagation_deltas[*].node_id` 同一编号空间）。 |
| `name`             | rotation 节点名（来自 ① `stages[k].nodes[*].name`）。 |
| `after_cut_point`  | 紧邻 N 之前的 rescale cut point 索引 `k`（即 `stage_anchor`）。 |
| `sf`               | N 实际运行时的 scaling factor，等于 `cut_point_sf[k].sf_post`。 |
| `count`            | N 的重复次数；总 rotation 数应按 `Σ count` 统计。 |

> 列表里只放 effective 的 rotation。Ineffective rotation **不会出现** 在
> 这个数组里，也不需要出现 —— 它们的 sf 已经能从对应 stage 的
> `cut_point_sf[k].sf` 推得（同 stage 的中间节点共用一个 sf）。

`effective_rotations: []` 的含义：该 block 在当前 skeleton 下没有任何
rotation 紧跟 rescale 执行。

---

## 7. 走读两个真实例子

### 例 1：`block5_n2`（mrpc）— 1 effective + 1 ineffective

config 拓扑（`stage` 编号 = `(c_k, c_{k+1}]`）：

```
SOURCE x_mean
stage 0:  [no rotations]   cut_point c_1 = ctct_xmean_over_std (CTCT, x2)
stage 1:  [no rotations]   cut_point c_2 = ctpt_gamal          (CTPT, +20)
stage 2:  rot_bs_wffn1     cut_point c_3 = ctpt_wffn1          (CTPT, +22)
stage 3:  rot_gs_wffn1     cut_point c_4 = ctct_gelu_x2        (CTCT, x2)
stage 4:  [no rotations]   cut_point c_5 = ctpt_gelu_coeff     (CTPT, +20)
```

batch 优化结果：

```
skeleton          = [0, 1, 3, 5, 6]
rescale 点集合     = skeleton[1:-1] = {1, 3, 5}
cut_point_sf      = [
  c_0 SOURCE  sf=31,
  c_1 CTCT    sf_pre=62  sf_post=31  drop=31,   ← rescale
  c_2 CTPT    sf=51,                            (非 rescale)
  c_3 CTPT    sf_pre=73  sf_post=31  drop=42,   ← rescale
  c_4 CTCT    sf=62,                            (非 rescale)
  c_5 CTPT    sf_pre=82  sf_post=31  drop=51    ← rescale
]
```

判定两条 rotation：

- `rot_bs_wffn1.stage_anchor = 2`。`2 ∉ {1, 3, 5}` → **ineffective**，
  不出现在输出里；它实际跑在 sf=51（= `cut_point_sf[2].sf`）。
- `rot_gs_wffn1.stage_anchor = 3`。`3 ∈ {1, 3, 5}` → **effective**。
  对应 `r = 2`（`s_2 = 3`），sf = `cut_point_sf[3].sf_post = 31`。

输出：

```json
"effective_rotations": [
  {"node_id": 5, "name": "rot_gs_wffn1", "after_cut_point": 3, "sf": 31, "count": 1}
]
```

### 例 2：`block1_mrpc` — 全部 ineffective

config 拓扑：

```
SOURCE gelu_out
stage 0:  bs_rot_in_mul                    cut_point c_1 = ctpt_ffn2     (CTPT, +20)
stage 1:  gs_rot_in_mul, rot_sum1 (×3)     cut_point c_2 = ctpt_inv_d_1  (CTPT, +20)
stage 2:  [no rotations]                   cut_point c_3 = ctct_ext_square (CTCT)
stage 3:  rot_sum2 (×3)                    cut_point c_4 = ctpt_inv_d_2  (CTPT, +20)
```

batch 优化结果：`skeleton = [0, 2, 4, 5]`，rescale 点 `{2, 4}`。

| rotation         | stage_anchor | ∈ {2, 4}? | effective? | 实际 sf                                         |
| ---------------- | ------------ | --------- | ---------- | ----------------------------------------------- |
| `bs_rot_in_mul`  | 0            | 否        | 否         | `cut_point_sf[0].sf = 30`（紧跟 SOURCE）        |
| `gs_rot_in_mul`  | 1            | 否        | 否         | `cut_point_sf[1].sf = 50`（紧跟非 rescale c_1） |
| `rot_sum1`       | 1            | 否        | 否         | `cut_point_sf[1].sf = 50`（同上）               |
| `rot_sum2`       | 3            | 否        | 否         | `cut_point_sf[3].sf = 68`（紧跟非 rescale c_3） |

所以 `effective_rotations: []`。这 4 个 rotation 全部跑在较高 sf 上 —
也意味着，如果想让某些 rotation 在低 sf 跑，需要重新选 skeleton 让 c_k
（rotation 上游的 cut point）落入 rescale 集合。

---

## 8. 怎么主动增加 effective rotation 数量

唯一的杠杆是 **改 skeleton**：让 ROTATION 的 `stage_anchor = k` 上的
那个 cut point `c_k` 被选为 rescale 点。常见两种做法：

1. **重跑 batch**：调整 `cost_params` / `q_legal_*` / `amplitude_budgets` /
   `snr_requirements` 让优化器在含 rotation 的 stage 末端更倾向 rescale。
2. **手动 replan**（推荐）：通过 `replan_what_if.py` 给定新的 `t_new`，
   触发 fusion-tolerant feasibility 推出新的 chain。新 chain 的
   `effective_rotations` 字段会立刻反映出来。

例如对 `block1_mrpc`：原 skeleton `[0, 2, 4, 5]`，rotation 都不在 rescale
点之后。如果通过 replan 让 c_3（`ctct_ext_square`）也变成 rescale 点，
则 `rot_sum2`（`stage_anchor=3`）就会进入 effective 列表。
是否真能成功取决于 amplitude / q_legal 是否容忍 c_3 的 q'（可能要靠
fusion）。

---

## 9. 与 cost / 噪声的关系（可选阅读）

- 优化器的 cost 函数：`weighted_cost = count * (cost_slope * level + cost_intercept)`，
  `level` 是节点执行时剩余的模数 level。Effective rotation 的 level 比
  其前一次 rescale 之前少 1（因为 rescale 消耗了一层），所以 cost 略低。
  对单个 rotation 而言差距很小，但 `count` 大的 rotation（如 `rot_sum2: count=3`）
  累积起来值得关心。
- KS 噪声方面：rotation 的 KS 噪声在 SEAL 中正比于当前模数；effective
  rotation 在 rescale 后执行，模数立即少一个 prime，KS 引入的相对噪声
  小一些。这部分通过 `noise_table.rescale[sf]` 反映在 amplitude /
  validator 阶段。

> 简言之：**effective_rotations 数 / 总 rotations 数** 可以视作优化器
> 把 “rescale 放在哪” 这件事是否 “对 rotation 友好” 的一个简单度量。
> 对于以 rotation-heavy 的 block（如 attention / softmax / 求和归并）
> 这个比值越高越好。
