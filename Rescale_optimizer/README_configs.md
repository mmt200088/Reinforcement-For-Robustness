# Rescale Optimizer — 配置 / 输出文件 & Replan 使用说明

本文档统一解释 4 类 JSON 文件的字段含义，以及 `scripts/replan_what_if.py`
的 “what-if 重规划” 用法。涉及到的目录布局：

```
Rescale_optimizer/
├── configs/
│   ├── wnli/                              # profile = wnli
│   │   ├── block1_wnli.json               # ① 输入：图配置
│   │   ├── ...
│   │   └── static_skeletons_wnli.json     # ② 输出：批量优化后的 skeleton + chain
│   └── mrpc/                              # profile = mrpc
│       ├── block1_mrpc.json
│       ├── ...
│       └── static_skeletons_mrpc.json
├── replan_configs/
│   ├── run_replan_batch.sh
│   ├── wnli/
│   │   ├── replan_actions_block1_wnli.json   # ③ 输入：用户动作 (t_new + 可选 deltas)
│   │   ├── ...
│   │   └── replan_block1_wnli.json           # ④ 输出：replan 结果
│   └── mrpc/  ...
├── scripts/
│   ├── batch_run_configs.py        # 生成 ②
│   ├── update_noise_tables_from_csv.py
│   ├── gen_replan_actions.py       # 生成 ③ 模板
│   └── replan_what_if.py           # ③ → ④
└── rescale_optimizer/              # 算法核心
```

下面逐个解释。

---

## ① 输入图配置 — `configs/<profile>/block*.json`

定义一个 “block” 的计算图（cut points + stages + nodes），以及噪声/精度
要求。`scripts/batch_run_configs.py` / `replan_what_if.py` 都吃这个文件。
以 `configs/mrpc/block3_exp_n2.json` 为例。

### 顶层字段

| key                  | 类型        | 含义 |
| -------------------- | ----------- | --- |
| `_description`       | 字符串      | 人读的图说明，可选。 |
| `global`             | 对象        | 全局参数 (h_sf / Q_legal)。 |
| `optimization`       | 对象        | 优化器超参 (cost / q_head / q_tail / 搜索预算)。 |
| `_noise_table_doc`   | 字符串      | noise_table 的来源说明，自动维护。 |
| `noise_table`        | 对象        | sf → noise 的查表，按 `op_type` 分组。 |
| `defaults`           | 对象        | 节点默认 op_type 等。 |
| `amplitude_budgets`  | int 数组    | 每个 cut point 的 amplitude budget（bit）。长度 = M+1。 |
| `amplitude_profiles` | 数组        | 每个 cut point 的 amplitude 分布。长度 = M+1。 |
| `snr_requirements`   | 数组        | 每个 cut point 的 SNR 要求。长度 = M+1。 |
| `source`             | 对象        | c₀ 节点（输入密文）的描述。 |
| `stages`             | 数组        | 后续 M 个 stage：每个 stage 包含中间 nodes 和一个 cut_point。 |
| `dummy_sink`         | 对象        | 虚拟 sink c_{M+1}。 |

### `global`

```json
"global": { "h_sf": 2, "q_legal_min": 30, "q_legal_max": 60 }
```

- `h_sf`：headroom scaling factor (bit)，每个 cut point 的工作 sf 至少为
  `t_base + h_sf`，给数值噪声留余量。
- `q_legal_min` / `q_legal_max`：单个 q' 允许的位宽下界 / 上界（部署到 SEAL
  的物理约束）。

### `optimization`

```json
"optimization": {
  "cost_params": { "lambda_0": 1.0, "lambda_1": 0.1, "alpha": 1.0, "beta": 0.05 },
  "q_head_bits": 60,
  "q_tail_bits": 60,
  "max_best_first_expansions": 64
}
```

- `cost_params`：成本函数权重，用于 backward DP 时打分。
- `q_head_bits` / `q_tail_bits`：固定的链头/链尾 prime 位宽（一般都是 60）。
- `max_best_first_expansions`：best-first 搜索的最大扩展数。

### `noise_table`

由 `scripts/update_noise_tables_from_csv.py` 从
`noise_inf_table.csv` 自动生成：

```json
"noise_table": {
  "rescale": { "12": 4.47e+0, "13": 2.24e+0, "14": 1.12e+0, ... , "60": 1.59e-14 },
  "fresh":   { "12": 4.49e+0, ...                                              }
}
```

- key 是 sf bit（默认 12..60，**逐 bit**），value 是噪声上界。
- `rescale` ← CSV 的 `B_rs`（rescale 后引入的额外噪声）。
- `fresh` ← CSV 的 `B_fresh`（一次新鲜密文的编码噪声）。
- `op_type` 在每个节点 / `defaults` 里指定，比如 source 用 `fresh`、其它用 `rescale`。
- N（多项式度数）由 base 名决定：`block1`、`block3_exp_n2`、`block5_n1` 用 N=8192，其余 N=16384。

### `amplitude_budgets` / `amplitude_profiles` / `snr_requirements`

三个长度 = `M+1` 的并列向量，第 `i` 个元素描述 cut point cᵢ 的精度需求：

```json
"amplitude_budgets":  [10, 10, 10, 10, 10, 10, 10],
"amplitude_profiles": [
  { "percentiles": [0.1, 0.5, 0.99], "values": [1e-4, 1e0, 1e3] },
  ...
],
"snr_requirements": [
  { "percentile": 0.8, "max_relative_error": 0.01 },
  ...
]
```

- `amplitude_budgets[i]`：cᵢ 处允许的 amplitude 增长 bit 数。
- `amplitude_profiles[i]`：cᵢ 处明文绝对值的分布（用于 `FindMinSF`）。
- `snr_requirements[i]`：cᵢ 处的 SNR 要求 `(分位数, 最大相对误差)`。
  优化器会选最小的 `sf` 让 `noise(sf) / |x|_p ≤ max_relative_error`。

### `source` 与 `stages[*].cut_point` / `stages[*].nodes`

每个 cut point / 中间 node 对象的常用字段：

| key                | 含义 |
| ------------------ | --- |
| `name`             | 节点名（用于 propagation_deltas 引用 / replan 中的 delta_overrides）。 |
| `type`             | `SOURCE` / `CTPT_MUL` / `CTCT_MUL` / `ROTATION` / `DUMMY_SINK`。 |
| `scale_delta_bits` | propagation delta：`CTPT_MUL` 表示 “乘上 sf=delta 的明文”，scale 累加 +delta；`CTCT_MUL` 通常被忽略（symmetric 时按 s→2s）；`ROTATION` 不动 scale。 |
| `count`            | 同类节点重复次数（例如 4 次 rotation）。 |
| `cost_slope`       | cost 函数对 modulus level 的线性系数。 |
| `cost_intercept`   | cost 函数的截距。 |
| `op_type`          | 在 `noise_table` 里查的 op 名（默认从 `defaults.op_type` 继承）。 |

CTCT 的 propagation 规则：

- **symmetric**（自乘 / 与同 scale 的另一个 ct 相乘）：`s → 2s`。
- **asymmetric**（与外部固定 scale 的 ct 相乘）：`s → s + other_ct_scale_bits`，
  通过把 cut_point 的 `scale_delta_bits` 显式设成 `other_ct_scale_bits`、
  并在 `_comment` 里写明 `other_ct_scale_bits=...` 表达。

### `dummy_sink`

```json
"dummy_sink": { "name": "dummy_sink", "amplitude_budget_bits": 0 }
```

- 算法内部用的虚拟节点 c_{M+1}，给 sink 留 0 bit budget 表示 "末端不再消费"。

---

## ② 批量优化结果 — `configs/<profile>/static_skeletons_<profile>.json`

由 `scripts/batch_run_configs.py` 生成，对该 profile 下每个 `block*.json`
跑一次 `optimize_rescale` 并把结果汇总。**这是真正可部署的 skeleton + 模数链的来源**。

### 顶层

```json
{
  "schema_version": 2,
  "generated_by": "scripts/batch_run_configs.py",
  "n_configs": 11,
  "n_success": 10,
  "results": [ ... ]
}
```

`results` 数组里每个元素对应一个 config。

### 每个 result（成功）

```json
{
  "config_name": "block3_exp_n2",
  "success": true,
  "skeleton": [0, 2, 3, 4],
  "cut_point_sf": [
    {"i": 0, "name": "X",             "type": "SOURCE",   "sf": 27},
    {"i": 1, "name": "ctpt_inv_2n",   "type": "CTPT_MUL", "sf": 43},
    {"i": 2, "name": "ctct_square_1", "type": "CTCT_MUL", "sf_pre": 86, "sf_post": 34, "drop": 52},
    {"i": 3, "name": "ctct_square_2", "type": "CTCT_MUL", "sf_pre": 68, "sf_post": 34, "drop": 34}
  ],
  "propagation_deltas": [
    {"node_id": 1, "name": "ctpt_inv_2n",   "type": "CTPT_MUL", "delta": 16},
    {"node_id": 2, "name": "ctct_square_1", "type": "CTCT_MUL", "delta": "x2"},
    {"node_id": 3, "name": "ctct_square_2", "type": "CTCT_MUL", "delta": "x2"}
  ],
  "modulus_chain": {
    "drop_order": [60, 52, 34, 60],
    "seal_order": [60, 34, 52, 60],
    "total_bits": 206
  },
  "effective_rotations": [
    {"node_id": 3, "name": "rot_bs_wffn1", "after_cut_point": 2, "sf": 30, "count": 1}
  ]
}
```

| key                  | 含义 |
| -------------------- | --- |
| `config_name`        | 配置文件 stem（比如 `block3_exp_n2`）。 |
| `success`            | 优化器是否找到可行链。 |
| `skeleton`           | 把 rescale 操作放在哪些 cut point 上：`[c_0=SOURCE, s_1, ..., s_R, c_{M+1}=DUMMY_SINK]`。前 `R+1` 项 = `[source, R 次 rescale 的 cut-point 索引]`，最后一项是虚拟 dummy_sink。`R = len(skeleton) - 2`，**rescale cut point 索引集合 = `skeleton[1:-1]`**。 |
| `cut_point_sf`       | 每个 cut point 在最终方案下的 scaling factor（bit）。 |
| `cut_point_sf[i].sf` | 该点没有 rescale 时的工作 scale。 |
| `cut_point_sf[i].sf_pre` / `sf_post` / `drop` | 该点 **要 rescale**，rescale 前 scale = `sf_pre`，rescale 掉 `drop` bit，rescale 后 scale = `sf_post`。即 `sf_post = sf_pre - drop`。 |
| `propagation_deltas` | 所有乘法 node 当前生效的 propagation delta。`CTPT_MUL` 是 int（明文 scale），`CTCT_MUL` 是 `"x2"`（symmetric）或 int（asymmetric 时的 other_ct_scale_bits）。`node_id` 与 graph 里的 node 一一对应。 |
| `modulus_chain.drop_order` | 优化器视角的 q 序列：`[q_head, q_1, ..., q_R, q_tail]`，`q_r` 是第 `r` 次 rescale 时丢掉的 prime 位宽。 |
| `modulus_chain.seal_order` | SEAL/部署视角的 q 序列：`[q_head, reverse(q_1..q_R), q_tail]`，因为 SEAL 从链尾向链头丢 prime。 |
| `modulus_chain.total_bits` | 整条链总 bit 数。 |
| `effective_rotations` | **紧跟 rescale 之后执行的 rotation**（详见下方）。空数组表示该 block 没有这种"低 sf 的 rotation"。 |

#### `effective_rotations` 字段

判定一个 ROTATION 节点 N 是否 effective：

1. 取 N 的 `stage_anchor = k`（即 N 落在 `(c_k, c_{k+1}]`）。
2. 若 `k ∈ skeleton[1:-1]`（即 c_k 是某次 rescale 的 cut point），N 就是 effective 的。
3. 否则（c_k 是 SOURCE / 非 rescale 乘法 cut point）N 是 ineffective，**不输出**。

物理含义：cut point 只出现在 stage 分界处，所以 N 与 c_k 之间在拓扑序里
**没有任何其它 cut point**，只可能夹一些不改 scale 的 rotation / pt 叶
节点。N 在 c_k 的 rescale 立刻执行后跑，sf 被压低到 `sf_post(c_k)`，所以
其工作 sf = `c_k.sf_post`，是当前位置上能拿到的最低 sf。

字段：

| key                | 含义 |
| ------------------ | --- |
| `node_id`          | graph 全局节点 id（与 `propagation_deltas[*].node_id` 同一编号空间）。 |
| `name`             | rotation 节点名（来自 ① 配置文件的 `stages[k].nodes[*].name`）。 |
| `after_cut_point`  | 紧邻其前的 rescale cut point 索引 `k`。 |
| `sf`               | 该 rotation 实际运行时的 scaling factor，等于 `cut_point_sf[k].sf_post`。 |
| `count`            | 该节点的重复次数（与 cost 公式里的 `count` 一致）。 |

> 反过来，如果一个 rotation 与上游 rescale 之间夹着至少一个 **非 rescale** 乘法
> cut point，它的 sf 已经被这些非 rescale 乘法累加放大，故被认为是 ineffective
> rotation。这种 rotation 的实际 sf 可以从 `cut_point_sf[k].sf` 推得（同样在
> 其 stage 内的中间节点也是同一个 sf），不放进 `effective_rotations`。

### 每个 result（失败）

```json
{
  "config_name": "block3_exp_n5",
  "success": false,
  "message": "Reachability: no path from c_0 to c_{M+1}. ..."
}
```

`message` 是失败原因；通常是 amplitude / q_legal 太紧导致没可行链。

### 重新生成

```bash
cd Rescale_optimizer

# 单独跑 mrpc
python scripts/batch_run_configs.py \
    --configs-dir configs/mrpc \
    --out configs/mrpc/static_skeletons_mrpc.json

# 同时跑 wnli
python scripts/batch_run_configs.py \
    --configs-dir configs/wnli \
    --out configs/wnli/static_skeletons_wnli.json

# 限定只跑某几个 config
python scripts/batch_run_configs.py \
    --configs-dir configs/mrpc \
    --configs block1_mrpc block5_n1 \
    --out /tmp/sub.json
```

---

## ③ Replan 输入动作 — `replan_configs/<profile>/replan_actions_<name>.json`

“What-if” 重规划的入口：在 ② 的 baseline 之上，给优化器手动指定
**每个 stage 的新 sf** 和 **每个乘法节点的新 propagation delta**，
让它在保持 skeleton 不变的前提下重算 chain（必要时融合 prime）。

### 字段

```json
{
  "config_name": "block3_exp_n2",
  "notes": "Auto-generated ...",
  "t_new": [30, 34, 34],
  "delta_overrides": {
    "ctpt_inv_2n":   16,
    "ctct_square_1": "x2",
    "ctct_square_2": "x2"
  }
}
```

| key               | 含义 |
| ----------------- | --- |
| `config_name`     | 对应的 graph config stem，需要与 ② 中能找到的 `config_name` 一致。 |
| `notes`           | 自由文本说明，可选。 |
| `t_new`           | 新的 “每个 skeleton 阶段的工作 sf”。长度 = `R + 1 = len(skeleton) - 1`：第 0 项是 source 的 sf，第 r 项是第 r 次 rescale 之后的 sf_post。 |
| `delta_overrides` | 节点名 → 新 delta：`CTPT_MUL` 给 int (= 明文 sf)，`CTCT_MUL` 给 `"x2"`（symmetric）或 int（asymmetric 时的 other_ct_scale_bits）。**只覆盖你想动的节点，其它保持 graph 原值**。 |

也支持 `propagation_deltas: [{name, delta}, ...]` 的形式（更接近 ② 的 schema）。

### 自动生成模板

```bash
# 默认：从 configs/static_skeletons.json 扫，写到旁边
python scripts/gen_replan_actions.py \
    --archive configs/mrpc/static_skeletons_mrpc.json \
    --out-dir replan_configs/mrpc \
    --filter "block*"
```

生成的模板里 `t_new` / `delta_overrides` 都会被预填成 baseline 值，
所以 “直接跑 = 恒等映射”，可以从这个出发逐项调整。

---

## ④ Replan 输出 — `replan_configs/<profile>/replan_<name>.json`

`scripts/replan_what_if.py --out` 写出的全量结果。结构：

```json
{
  "config_path": "/abs/path/to/configs/mrpc/block3_exp_n2.json",
  "config_name": "block3_exp_n2",
  "valid": true,
  "fusion_count": 0,
  "actions_file": "/abs/path/to/replan_actions_block3_exp_n2.json",

  "baseline": {
    "skeleton": [0, 2, 3, 4],
    "t_baseline": [30, 34, 34],
    "q_bits_baseline": [58, 34]
  },

  "t_new": [30, 34, 34],
  "delta_overrides": {"ctpt_inv_2n": 16, "ctct_square_1": "x2", "ctct_square_2": "x2"},

  "result": {
    "valid": true,
    "message": "replan OK after 0 fusion(s). R: 2 -> 2.",
    "fusion_count": 0,
    "skeleton": [0, 2, 3, 4],
    "q_initial": [58, 34],
    "q_final":   [58, 34],
    "t_final":   [30, 34, 34],
    "delta_q_vs_baseline": [0, 0],
    "applied_delta_overrides": {"ctpt_inv_2n": 16, "ctct_square_1": "x2", "ctct_square_2": "x2"},
    "fusions": [],
    "chain": {
      "q_head_bits": 60, "q_bits": [58, 34], "q_tail_bits": 60,
      "total_bits": 212, "R": 2
    },
    "invalid_chain": null
  },

  "new_compact_config": { ... 与 ② 中单条 result 同 schema ... }
}
```

| key                                       | 含义 |
| ----------------------------------------- | --- |
| `valid`                                   | replan 后是否得到合法 chain。 |
| `fusion_count`                            | 触发了多少次 prime 融合（小 q 与相邻 q 合并）。 |
| `baseline.skeleton/t_baseline/q_bits_baseline` | 来自 ②：原始 skeleton / 原始 t / 原始 q（不含 head/tail）。 |
| `t_new` / `delta_overrides`               | 实际生效的 ③ 内容（合并 CLI / file 之后）。 |
| `result.q_initial`                        | 用 `t_new` + delta 直接传播得到的 raw q 序列（融合前）。 |
| `result.q_final`                          | 经过 fusion-tolerant feasibility 修复后的 q 序列。 |
| `result.t_final`                          | 与 `q_final` 对齐的、chain-consistent 的最终 t 序列（即融合后 SEAL 真实会看到的 sf）。 |
| `result.delta_q_vs_baseline`              | `q_final - q_bits_baseline`（按位差，有 fusion 时长度可能短）。 |
| `result.applied_delta_overrides`          | 实际写到 graph 的 delta（可能是 ③ 给的全部 / 部分）。 |
| `result.fusions`                          | 每次融合事件：`fused_position` (被去掉的 q 位置) / `fused_into` / `small_q` / `neighbour_q_before` / `neighbour_q_after`。 |
| `result.chain`                            | 最终合法 chain。`q_bits` 是 `[q_1..q_R]`，加上 head/tail 就是 `drop_order`。 |
| `result.invalid_chain`                    | 仅当 `valid=false` 时填，给出违规链以便诊断。 |
| `new_compact_config`                      | 把 `result` 整理成 ② 单条 result 的同 schema，便于后续直接喂回部署/再 replan。包含 `cut_point_sf` / `propagation_deltas` / `modulus_chain` / `effective_rotations` / `fusion_count`。其中 `effective_rotations` 的语义与 ② 完全一致（紧跟 rescale 的 rotation，sf = `cut_point_sf[k].sf_post`）。 |

### Fusion-tolerant 可行性（重要）

输入的 `t_new` 经过 propagation 算出的某个 `q'_r` 可能 < `q_legal_min`（如 30 bit），
此时不直接判失败，而是按下面规则尝试合并：

1. 找相邻 `q'_{r-1}` 或 `q'_{r+1}`，若 `q'_r + 邻居 ≤ q_legal_max`（如 60 bit），
   就把 `q'_r` 处的 rescale 步骤去掉，邻居 `q'` 的位宽吸收 `q'_r`。
2. 删除一个 rescale 后，整链重新走一遍 `propagate_scale`，更新所有
   下游的 `t`、`q'`，因为去掉一次 rescale 会让后面的 sf 累加更高。
3. 上述过程递归直到所有 q 都落在 `[q_legal_min, q_legal_max]`，或没有可融的 q'。
4. 仍存在 `q' < q_legal_min` ⇒ 报 `valid=false`，附 `invalid_chain` 供调试。

`fusion_count` = 整个过程发生过的合并次数。

---

## ⑤ 怎么用 `replan_what_if.py`

### 单个跑

```bash
cd Rescale_optimizer

python scripts/replan_what_if.py \
    --config configs/mrpc/block3_exp_n2.json \
    --baseline-from configs/mrpc/static_skeletons_mrpc.json \
    --actions-file replan_configs/mrpc/replan_actions_block3_exp_n2.json \
    --out replan_configs/mrpc/replan_block3_exp_n2.json
```

执行流程：

1. 加载 `--config` 的图，建 feasibility DAG（不走优化，只是要 stage 信息）。
2. 从 `--baseline-from`（② 的 archive）里按 `config_name`（默认取 `--config` 的 stem）
   找到 baseline 的 `skeleton` / `t_baseline` / `q_bits_baseline`。
3. 从 `--actions-file` 读 `t_new` 和 `delta_overrides`。
4. 把 delta_overrides 写到 graph 的乘法 nodes 上，按 `t_new` 重新做 propagate / 算 q。
5. 跑 fusion-tolerant 可行性修复。
6. 终端打印一段 `result.summary()`；若提供 `--out` 还会把 `④` 全量结果落盘。

> 进程**不会**改 `--config` 也**不会**改 `--actions-file`，所有改动都
> 只发生在内存里的 graph 上。

### 主要 CLI 参数

| 参数                       | 作用 |
| -------------------------- | --- |
| `--config`                 | 必填：图配置 ①。 |
| `--baseline-from`          | 与 `--skeleton` 二选一：从 ② archive 里挑 baseline。 |
| `--config-name`            | 与 `--baseline-from` 配合使用，覆盖默认的 “stem 命名”。 |
| `--skeleton i1 i2 ...`     | 与 `--baseline-from` 二选一：手动给 baseline skeleton。 |
| `--t-baseline a b ...`     | 手动模式下的 baseline t（仅用于 `delta_q` 报告，可省）。 |
| `--actions-file path.json` | 推荐：一个文件里同时给 `t_new` + `delta_overrides`（即 ③）。 |
| `--t-new a b ...`          | 直接命令行给 `t_new`（覆盖 `--actions-file` 的同名字段）。 |
| `--t-new-file path.json`   | 从 JSON 给 `t_new`（list 或 `{"t_new": [...]}`）。 |
| `--delta-override NAME V`  | 重复使用，逐个覆盖单个节点的 delta。 |
| `--delta-overrides-file`   | 从 JSON 给一组 delta_overrides（与 `--actions-file` 等价的子集）。 |
| `--out path.json`          | 把 ④ 写到 path。 |
| `--log-level`              | `WARNING` / `INFO` / `DEBUG`。 |

优先级：CLI 显式参数 > `--actions-file` 内字段。

### 三类常用场景

**A. 用 actions 文件做完整重规划（推荐）**

```bash
python scripts/replan_what_if.py \
    --config configs/mrpc/block1_mrpc.json \
    --baseline-from configs/mrpc/static_skeletons_mrpc.json \
    --actions-file replan_configs/mrpc/replan_actions_block1_mrpc.json \
    --out replan_configs/mrpc/replan_block1_mrpc.json
```

**B. 临时 what-if，只改一两个节点的 delta**

```bash
python scripts/replan_what_if.py \
    --config configs/mrpc/block1_mrpc.json \
    --baseline-from configs/mrpc/static_skeletons_mrpc.json \
    --t-new 30 34 34 38 \
    --delta-override ctpt_ffn2 18 \
    --delta-override ctct_ext_square x2
```

**C. 不依赖 archive，手动给 baseline**

```bash
python scripts/replan_what_if.py \
    --config configs/mrpc/block1_mrpc.json \
    --skeleton 0 2 4 5 \
    --t-baseline 30 34 34 34 \
    --t-new       32 34 34 36
```

### 一次跑一个 profile 的所有 config

```bash
# 默认：自动发现 replan_configs/ 下的所有 profile（mrpc, wnli, ...）
bash replan_configs/run_replan_batch.sh

# 指定 profile
bash replan_configs/run_replan_batch.sh mrpc
bash replan_configs/run_replan_batch.sh wnli mrpc
```

每个 `replan_actions_<stem>.json` 都会跑出一个旁边的
`replan_<stem>.json`（即 ④），并打印每条的 `fusion_count`。

### 退出码

| code | 含义 |
| ---- | --- |
| 0    | replan 成功（合法 chain）。 |
| 1    | 输入错误（找不到 config / 解析失败）。 |
| 3    | replan 跑完但 chain 不合法（`valid=false`）。 |

---

## 端到端工作流速查

```bash
cd Rescale_optimizer

# 0) (可选) 从最新实测 CSV 刷新所有 noise_table（1-bit 步长）
python scripts/update_noise_tables_from_csv.py

# 1) 跑批量优化，得到部署用 skeleton + chain
python scripts/batch_run_configs.py \
    --configs-dir configs/mrpc \
    --out configs/mrpc/static_skeletons_mrpc.json

# 2) 为每个 block 生成 replan actions 模板（默认是 baseline 恒等）
python scripts/gen_replan_actions.py \
    --archive configs/mrpc/static_skeletons_mrpc.json \
    --out-dir replan_configs/mrpc \
    --filter "block*"

# 3) 编辑 replan_configs/mrpc/replan_actions_*.json，调整 t_new / delta_overrides

# 4) 一次性跑所有 replan 并写 ④
bash replan_configs/run_replan_batch.sh mrpc

# 5) 查看每个 replan_<stem>.json 中的：
#    - new_compact_config.modulus_chain  → 部署用链
#    - fusion_count / fusions            → 是否触发 fusion
#    - delta_q_vs_baseline               → 与原 baseline 的 q 差
```
