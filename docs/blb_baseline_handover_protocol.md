# BLB Stage-2 RL ↔ Rescale_optimizer Baseline 握手

本文档涵盖两条 baseline 获取通路：

- **[推荐 / 当前实际使用] 路径 A — 直接读 static_skeletons archive**：
  RO 团队人工生成 `Rescale_optimizer/configs/<dataset>/static_skeletons_<dataset>.json`，
  RL 这边按 Stage-1 配置直接读取并组装。**见第 0 节**。
- **[保留向后兼容] 路径 B — 双向 JSON 握手协议 v1**：RL 写请求 → RO 处理 → 写响应。
  本文档其余部分（1-10 节）描述路径 B。

---

## 0. 路径 A：直接读 static_skeletons archive（推荐）

### 0.1 文件位置

```
Rescale_optimizer/configs/<dataset>/static_skeletons_<dataset>.json
```

例：mrpc 任务的归档在 `Rescale_optimizer/configs/mrpc/static_skeletons_mrpc.json`。
RO 团队负责手工维护这份文件，每条 entry 是一个 (block, [degree]) graph 的 baseline。

### 0.2 archive schema 摘要

```json
{
  "schema_version": "...",
  "results": [
    {
      "config_name": "block1_mrpc",   /* 或 block3_exp_n2 / block5_n4 等 */
      "success": true,
      "skeleton": [0, 2, 4, 5],
      "cut_point_sf": [
        {"i": 0, "name": "gelu_out", "type": "SOURCE", "sf": 30},           /* 第一项 = fresh */
        {"i": 1, "name": "ctpt_ffn2", "type": "CTPT_MUL", "sf": 50},        /* 中间项 = propagation 累计 sf */
        {"i": 2, "name": "ctpt_inv_d_1", "type": "CTPT_MUL",
         "sf_pre": 70, "sf_post": 34, "drop": 36}                           /* 带 sf_post = rescale */
      ],
      "propagation_deltas": [
        {"node_id": 2, "name": "ctpt_ffn2", "type": "CTPT_MUL", "delta": 20},      /* numeric delta = encode */
        {"node_id": 6, "name": "ctct_ext_square", "type": "CTCT_MUL", "delta": "x2"}  /* x2 = 不是 RL 动作 */
      ],
      "modulus_chain": {"drop_order": [...], "total_bits": 210, ...},
      "effective_rotations": [...]
    },
    ...
  ]
}
```

**字段语义**：

- `cut_point_sf[0]`（SOURCE）：对应 **fresh 噪声**，scaling_factor = `sf`。
- `cut_point_sf[i]` (i ≥ 1) 带 `sf_post`：对应 **rescale 噪声**，scaling_factor = `sf_post`。
- `cut_point_sf[i]` (i ≥ 1) 不带 `sf_post`：只是 propagation 累计 SF，**不对应任何 RL 动作**。
- `propagation_deltas` 带 numeric `delta`：对应 **encode 噪声**，scaling_factor = `delta`。
- `propagation_deltas` 带 `"x2"`：平方操作，**不对应 RL 动作**。
- `modulus_chain.drop_order`：所有动作选取后的模数链 bit 序列。
- `effective_rotations`：RO baseline 下启用的 rotation 候选；RL 通过 `apply_rotation_flags_to_cfg`
  反写到 cfg。

### 0.3 每层 graph_key 选择规则

逐层根据该层 Stage-1 (gelu_degree, softmax_degree) 拼出 `graph_key`：

| block | graph_key 规则 |
|---|---|
| 1 | `block1_<dataset>` |
| 2 | `block2_<dataset>` |
| 3 | `block3_exp_n<softmax_degree>` |
| 4 | `block4` |
| 5 | `block5_n<gelu_degree>` |

**注意**：(block=1, layer=0) **跳过** —— layer-0 没有上游 FFN2，第一个 HE 配置无损，
block 1 噪声整体不安装。所以 RL 实际取 `5 * num_layers - 1` 个 (block, layer) baseline。

### 0.4 RL 一侧 API

`blb_stage2_rl.baseline_bootstrap` 提供两个函数：

```python
from blb_stage2_rl.baseline_bootstrap import (
    load_static_skeletons_baseline,           # 读 JSON + 抽取
    static_skeletons_baseline_to_action,      # 抽取 → RL 动作向量 + MaxSFsTable
)

baseline = load_static_skeletons_baseline(
    rescale_optimizer_root="Rescale_optimizer",
    dataset="mrpc",
    num_layers=12,
    gelu_per_layer=[1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1],
    softmax_per_layer=[2, 2, 5, 5, 5, 2, 5, 2, 5, 5, 6, 2],
)
# baseline.per_block_layer[(block_idx, layer_idx)] = StaticSkeletonsLayerBlock(...)
# baseline.aggregate_total_bits / aggregate_fusion_count

action_vec, max_sfs, cost_stats, diagnostics = static_skeletons_baseline_to_action(baseline)
# action_vec:  np.ndarray, 长度 = sum(action_dims_for_config(num_layers))
# max_sfs:     校准过的 MaxSFsTable —— "max idx" 对应 JSON baseline SF
# cost_stats:  BaselineCostStats（可直接喂 reward.compute_reward）
# diagnostics: 调试信息（active/inactive slot 计数、未映射节点）
```

### 0.5 校准 MaxSFsTable 的语义

为什么要返回 MaxSFsTable？因为 RL 的动作 ladder 由 `sf_from(idx, max_sf, levels)` 决定，
而 JSON baseline 的 SF（如 rescale sf_post = 34）可能**超过** RL 默认的 `max_sf=22`。

校准做法：对于每个 RL 字段，如果 JSON 给了 baseline SF，则把对应的 `max_sf` 设为该 SF。
这样：

```
sf_from(max_idx, max_sf=baseline_sf, levels) == baseline_sf
```

转换器返回的 `baseline_action_vec` 是 RO 推荐 baseline；非 rescale 槽取 max_idx，
出现在 JSON `sf_post/drop` 里的 rescale 槽取 max_idx。

未在 JSON 出现的 rescale 字段（= "off at RO baseline"）会被写成 action index 0，
解码为 `None`，因此明文噪声模拟也不会安装该 rescale 噪声；同时这些槽位会列在
`diagnostics["inactive_rescale_slots"]`，便于审计。

### 0.6 用法示例：runner 启动前接入

```python
# rl_tune.py / BLBStage2RLRunner.run 入口附近
from blb_stage2_rl.baseline_bootstrap import (
    load_static_skeletons_baseline, static_skeletons_baseline_to_action,
)

# Stage-1 完成后拿到 fixed_gelu / fixed_softmax
baseline = load_static_skeletons_baseline(
    rescale_optimizer_root=train_cfg.inproc_rescale_optimizer_root,
    dataset=train_cfg.profile,
    num_layers=int(ev.total_layers),
    gelu_per_layer=fixed_gelu,
    softmax_per_layer=fixed_softmax,
)
baseline_action_vec, max_sfs_table, baseline_cost_stats, diag = \
    static_skeletons_baseline_to_action(baseline)

# 把 max_sfs_table 装到 env 里替代 load_max_sfs(...) 的默认表
env.max_sfs = max_sfs_table

# 用 baseline_action_vec 作为 warmstart anchor
# 用 baseline_cost_stats 作为 RewardWeights 校准的输入
```

### 0.7 错误处理

- `FileNotFoundError`: archive 路径不存在 → 提示用户克隆 Rescale_optimizer 仓库。
- `BaselineHandoverError`: archive 里某 `graph_key` 缺失（例 `block5_n2` 没 success entry）。
- 在 BLB Stage-2 RL 训练入口，这两类错误都是硬错误：runner 直接停止，不再回退到
  `load_max_sfs(...)` 或 all-max 动作估计 baseline。
- archive 里某条 entry 的 SOURCE / propagation_deltas 出现未映射节点 → 不报错，挂
  `diagnostics["unmapped_nodes"]` 让 caller 知情。

### 0.8 与路径 B 的关系

路径 A 是当前推荐路径；路径 B 是更早设计的"双向 JSON 握手协议"。
两者**互斥使用**：

- 路径 A：archive 文件是单向 input（RL 只读，RO 写）。
- 路径 B：request/response 双向（RL 写请求，RO 写响应）。

路径 B 的 schema 仍然保留下来（下方 1-10 节），方便将来如需切换或自动化。

---

# 路径 B：BLB Stage-2 RL ↔ Rescale_optimizer Baseline 握手协议（JSON）

**版本**：`v1` — 2026-05-10。

本协议定义 Stage-1 完成后、Stage-2 RL 训练启动前的**一次性 baseline 握手**：

1. **RL 一侧**（本仓库）把 Stage-1 RL 选出的每层 GELU / Softmax degree 写入一份
   请求 JSON。
2. **Rescale_optimizer 一侧**（合作者维护）按照该请求逐 (block, layer) 计算 BLB
   Stage-2 的全 max baseline，验证模数链合法性，把结果写入响应 JSON。
3. RL 一侧读取响应 JSON，转换成 `blb_stage2_rl.reward.BaselineCostStats` 以及
   per-(block, layer) baseline cfg，进入 Stage-2 训练循环。

> **训练循环不走 JSON。**Stage-2 训练**每一步**都通过
> `rescale_optimizer_bridge.InProcessInvoker` 直接 `import rescale_optimizer`
> 调 `replan_with_user_actions`（ms 级）；JSON 握手只在 baseline 阶段执行一次。

读者：BLB RL 这边的同事 + Rescale_optimizer 这边的同事。两边都按本文档实现。

---

## 1. 文件位置约定

两个 JSON 文件都放在仓库根目录下的 `handover/` 目录里（如果不存在，写方负责
`mkdir -p`）：

```
<repo_root>/handover/baseline_request_<dataset>.json   # RL 写
<repo_root>/handover/baseline_response_<dataset>.json  # RO 写
```

`<dataset>` 是 GLUE 任务名（`mrpc` / `cola` / `rte` / `stsb` 等），与
`scripts/blb_export_action_registry.py --profile <dataset>` 的 profile 一致。

如果存在多个并发 baseline 计算需求（罕见），可在文件名里加 `_<request_id>` 后缀。

---

## 2. 请求 schema：`baseline_request_<dataset>.json`

```json
{
  "schema": "blb_baseline_request_v1",
  "request_id": "20260510-mrpc-bert-base",
  "dataset": "mrpc",
  "model": "bert-base",
  "num_layers": 12,
  "stage1_config": {
    "gelu_degree_per_layer":    [1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1],
    "softmax_degree_per_layer": [2, 2, 5, 5, 5, 2, 5, 2, 5, 5, 6, 2]
  },
  "rl_max_sfs": {
    "block1": {"ctpt_ffn2": 22, "ctpt_inv_d_1": 22, "ctpt_inv_d_2": 22},
    "block2": {"ctpt_gama1": 22, "ctpt_wq_wk": 22, "ctpt_rotKT_mask1": 22,
               "ctpt_rotKT_mask2": 22, "ctpt_mask": 22},
    "block3": {"ctpt_inv_2n": 22},
    "block4": {"ctpt_mask2": 22, "ctpt_mask": 22, "ctpt_wo_attnout": 22,
               "ctpt_inv_d_1": 22, "ctpt_inv_d_2": 22},
    "block5": {"ctpt_gamal": 22, "ctpt_wffn1": 22, "ctpt_gelu_coeff": 22}
  },
  "rl_max_fresh_sfs": {
    "block1.gelu_out_fresh": 30,
    "block2.inv_std_fresh": 30,
    "block2.x_centered_fresh": 30,
    "block3.x_fresh": 30,
    "block4.softmax_out_fresh": 30,
    "block4.v_fresh": 30,
    "block5.inv_std_fresh": 30,
    "block5.x_centered_fresh": 30,
    "first_input_fresh": 30
  },
  "rl_max_truncation_k": 13,
  "blb_first_input_N": 8192,
  "rl_repo_commit": "abcdef1",
  "generated_at": "2026-05-10T14:30:00+08:00",
  "generator": "blb_stage2_rl.baseline_bootstrap.write_baseline_request"
}
```

### 字段语义

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `schema` | str | ✓ | 永远固定 `"blb_baseline_request_v1"`；下一版改 `_v2`。 |
| `request_id` | str | ✓ | 唯一标识，便于多请求并存 / 错误溯源；推荐 `<date>-<dataset>-<model>` 格式。 |
| `dataset` | str | ✓ | GLUE 任务名（影响 `block1_<dataset>` / `block2_<dataset>` graph）。 |
| `model` | str | ✓ | 例：`bert-base`；用于 RO 端选层数 / 嵌入维度等。 |
| `num_layers` | int | ✓ | encoder 层数，与 `gelu_degree_per_layer` / `softmax_degree_per_layer` 长度必须相等。 |
| `stage1_config.gelu_degree_per_layer` | list[int] | ✓ | 长度 `num_layers`；元素 ∈ {1, 2, 4}（block5 graph 命名 `block5_n<deg>`）。 |
| `stage1_config.softmax_degree_per_layer` | list[int] | ✓ | 长度 `num_layers`；元素 ∈ {2, 3, 4, 5, 6}（block3 graph 命名 `block3_exp_n<deg>`）。 |
| `rl_max_sfs` | dict | 可选 | per-block per-node 的 max SF 上限；不传则用 RO 自己的 `static_skeletons_<dataset>.json` 默认。 |
| `rl_max_fresh_sfs` | dict | 可选 | 每个 fresh / first-input fresh 的 max SF；扁平 `<block>.<field>: int`。 |
| `rl_max_truncation_k` | int | 可选 | RL 端的 max truncation k；RO 不参与计算 K，但回写到响应里方便 RL 一致性自检。默认 `13`。 |
| `blb_first_input_N` | int | 可选 | layer-0 input fresh 的 CKKS poly degree；默认 `8192`。 |
| `rl_repo_commit` | str | 可选 | RL 仓库当时的 git commit；用作错误调查标签。 |
| `generated_at` | str | 可选 | ISO-8601 时间戳。 |
| `generator` | str | 可选 | 写文件的代码符号；方便溯源。 |

### degree 取值范围与 graph 文件名约定

| Stage-1 字段 | 允许值 | 影响 |
|---|---|---|
| `gelu_degree_per_layer[i]` | 1, 2, 4 | block5 用 `block5_n<deg>.json`；deg=1 时 N=8192，否则 N=16384。 |
| `softmax_degree_per_layer[i]` | 2, 3, 4, 5, 6 | block3 用 `block3_exp_n<deg>.json`；deg=2 时 N=8192，否则 N=16384。 |

**block 1 / 2 / 4 与 degree 无关**：block1 / block2 用 `block{N}_<dataset>.json`，
block4 用 `block4.json`（共享所有 dataset）。

---

## 3. 响应 schema：`baseline_response_<dataset>.json`

```json
{
  "schema": "blb_baseline_response_v1",
  "request_id": "20260510-mrpc-bert-base",
  "request_path": "handover/baseline_request_mrpc.json",
  "dataset": "mrpc",
  "model": "bert-base",
  "num_layers": 12,
  "ok": true,
  "error": null,
  "ro_version": "1.2.0",
  "static_skeletons_archive": "Rescale_optimizer/configs/mrpc/static_skeletons_mrpc.json",
  "completed_at": "2026-05-10T14:31:00+08:00",
  "results": [
    {
      "config_name": "block1_mrpc_L0",
      "graph_key": "block1_mrpc",
      "block": 1,
      "layer": 0,
      "success": true,
      "skeleton": [0, 2, 4, 5],
      "t_baseline": [30, 22, 22, 0],
      "q_bits_baseline": [60, 60, 60],
      "modulus_chain": {
        "q_head_bits": 60,
        "q_bits": [60, 60, 60],
        "q_tail_bits": 60,
        "total_bits": 240,
        "R": 3
      },
      "fusion_count": 0,
      "invalid_chain": null,
      "cut_point_sf": [
        {"i": 0, "sf": 30, "node": "ctct_gelu_out"},
        {"i": 2, "sf_post": 22, "node": "ctct_mean_rescale"},
        {"i": 4, "sf_post": 22, "node": "ctct_var_rescale"},
        {"i": 5, "sf": 0, "node": "dummy_sink"}
      ],
      "effective_rotations": ["rot_after_gelu_out_fresh", "rot_after_var_result_rescale"],
      "graph_path": "Rescale_optimizer/configs/mrpc/block1_mrpc.json"
    },
    {
      "config_name": "block2_mrpc_L0",
      "block": 2,
      "layer": 0,
      "..."
    },
    {
      "config_name": "block3_exp_n2_L0",
      "graph_key": "block3_exp_n2",
      "block": 3,
      "layer": 0,
      "..."
    }
  ],
  "aggregate": {
    "valid_block_count": 60,
    "invalid_block_count": 0,
    "total_bits_sum": 14989,
    "total_fusion_count": 0
  },
  "warnings": [
    "block5_n4_L5: warning: graph fold x^3 into x^4 — gelu_power_rescales[1] not on skeleton"
  ]
}
```

### 顶层字段

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `schema` | str | ✓ | 永远 `"blb_baseline_response_v1"`。 |
| `request_id` | str | ✓ | 必须**逐字** echo 请求里的 `request_id`；RL 据此识别响应。 |
| `request_path` | str | 可选 | 响应所对应的请求文件路径（相对仓库根）。 |
| `dataset` / `model` / `num_layers` | 同请求 | ✓ | 直接 echo，便于响应自包含。 |
| `ok` | bool | ✓ | `true` ⇒ 至少所有必选 (block, layer) 都计算成功；`false` ⇒ 至少一处失败。 |
| `error` | str / null | ✓ | `ok==true` 时必须 null；`ok==false` 时给一段人类可读错误描述。 |
| `ro_version` | str | 可选 | RO 端代码版本号；便于错误调查。 |
| `static_skeletons_archive` | str | 可选 | RO 用了哪份 baseline 归档（相对路径）；用于调试一致性。 |
| `completed_at` | str | 可选 | ISO-8601 时间戳。 |
| `results` | list[dict] | ✓ | 每个 (block, layer) 一条记录，长度 `5 * num_layers - 1`。**注意：layer 0 的 block 1 整体不需要**（语义：layer-0 没有上游 FFN2，第一个 HE 配置无损，block1 噪声不安装）。所以 results 缺 (block=1, layer=0) 这一条；若 RO 仍返回它也兼容接收，但 RL 一侧不消费。 |
| `aggregate` | dict | ✓ | 跨 (block, layer) 聚合统计。 |
| `warnings` | list[str] | 可选 | 非致命警告，RL 端只 log。 |

### `results[i]` 字段

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `config_name` | str | ✓ | RL 端 `make_config_name(profile, block, layer, cfg)` 的输出，例 `block3_exp_n4_L5`。 |
| `graph_key` | str | ✓ | 剥掉 `_L<i>` 的 graph key，例 `block3_exp_n4`。 |
| `block` | int | ✓ | 1..5。 |
| `layer` | int | ✓ | 0..num_layers-1。 |
| `success` | bool | ✓ | 该 (block, layer) 的 baseline 计算是否成功。 |
| `skeleton` | list[int] | ✓（success=true 时） | `Rescale_optimizer.replan_with_user_actions` 用的 skeleton；末尾可含 `dummy_sink` (`graph.M+1`)。 |
| `t_baseline` | list[int] | ✓（success=true） | 长度同 skeleton；元素是每个 stage 的 baseline SF。 |
| `q_bits_baseline` | list[int] | ✓（success=true） | 长度 `R = len(skeleton) - 1`；元素是每段模数链的 bit 宽度（去掉首尾 head/tail prime）。 |
| `modulus_chain` | dict | ✓（success=true） | 与 `replan_what_if.py` 输出兼容：`q_head_bits` / `q_bits` / `q_tail_bits` / `total_bits` / `R`。 |
| `fusion_count` | int | ✓（success=true） | baseline 调用的 fusion 次数（通常 0）。 |
| `invalid_chain` | dict / null | ✓ | null = baseline 合法；dict = 不合法（含 `q_head_bits` / `q_bits` / `q_tail_bits` 的非法状态）。 |
| `cut_point_sf` | list[dict] | 可选 | 每个 stage 的 source SF / post SF；和 `static_skeletons_<dataset>.json` 同结构。 |
| `effective_rotations` | list[str] | 可选 | RO 在 baseline 下选定的 rotation flag 名（用 RL 命名空间），例 `"rot_after_gelu_out_fresh"`。 |
| `graph_path` | str | 可选 | 该 graph json 路径（相对仓库根）；调试用。 |

### `aggregate` 字段

| 字段 | 必填 | 说明 |
|---|---|---|
| `valid_block_count` | ✓ | success=true 的 (block, layer) 总数（**不含** layer-0 block 1，因为不发请求）。 |
| `invalid_block_count` | ✓ | success=false 的总数。 |
| `total_bits_sum` | ✓ | sum of `modulus_chain.total_bits` over all valid (block, layer)。 |
| `total_fusion_count` | ✓ | sum of `fusion_count`。 |

### 关于 (block=1, layer=0) 缺失

RL 端不再为 layer 0 安装 block1 噪声（与 Rescale_optimizer 习惯一致）。因此：

1. RL 不会发送 `block1_<dataset>_L0` 的请求。
2. RO 计算 baseline 时**可选**：跳过 `(block=1, layer=0)`，或者返回但 RL 会忽略。
   推荐 RO 也跳过该位置，避免无意义的资源消耗。
3. 对应的 RL action 槽位（layer 0 block1 的 9 个）仍存在于动作向量里以维持 policy
   网络 shape，但被 `_is_action_field_effective` 标记为 `effective=False`。
4. 第一个 HE 配置（layer-0 input）的 fresh 噪声同样**不**注入；
   `first_input_fresh` 槽位也是 `effective=False`，纯占位。

---

## 4. 失败语义

如果 baseline 算不出来（例如 RL 给的 Stage-1 (gelu, softmax) 组合在某层导致 graph
不存在 / chain 不合法），**RO 不要中断响应，而是把那条 (block, layer) 的 `success`
设为 `false`** 并填上 `invalid_chain` / `error_message`：

```json
{
  "config_name": "block5_n4_L5",
  "graph_key": "block5_n4",
  "block": 5,
  "layer": 5,
  "success": false,
  "invalid_chain": {
    "q_head_bits": 60,
    "q_bits": [60, 60, 25],
    "q_tail_bits": 60
  },
  "error_message": "min q_bit 25 < q_min=30 at stage 2; fusion cannot reduce."
}
```

顶层 `ok` 字段：`results` 中**任何一条** `success=false` ⇒ 顶层 `ok=false` +
`error` 给出 "n_failed/n_total blocks failed; first failure: ..."。

RL 一侧收到 `ok=false` 时**拒绝启动训练**（baseline 都过不了，全 max action 都
不合法，RL 没办法启动）。

---

## 5. 调用流程（双方协作）

```
[RL]                                    [RO]
  │
  ├─ stage 1 RL 完成
  │
  ├─ baseline_bootstrap.write_baseline_request(
  │       repo_root, dataset, stage1_gelu, stage1_softmax)
  │   ⇒ handover/baseline_request_<dataset>.json
  │
  ├─ 通知 RO（人工 / 文件 watcher / CI 触发器）
  │                                       │
  │                                       ├─ 读 handover/baseline_request_<dataset>.json
  │                                       ├─ 校验 schema, num_layers, degree 取值
  │                                       ├─ for each (block, layer):
  │                                       │     pick graph_key by (block, dataset, gelu_deg, softmax_deg)
  │                                       │     load graph from configs/<dataset>/<graph_key>.json
  │                                       │     compute baseline (skeleton, t_baseline, q_bits)
  │                                       │     run replan_with_user_actions(t_new=t_baseline, deltas={})
  │                                       │     verify modulus_chain valid
  │                                       ├─ 聚合 aggregate stats
  │                                       ├─ write handover/baseline_response_<dataset>.json
  │                                       │
  │                                       └─ 通知 RL（同上）
  │
  ├─ baseline_bootstrap.read_baseline_response(
  │       repo_root, dataset, expected_request_id)
  │   ⇒ BaselineHandoverResult
  │
  ├─ 转换成 BaselineCostStats + per-block-per-layer 全 max action
  │
  └─ 启动 BLB Stage-2 RL 训练（in-process invoker，无 JSON 流量）
```

**通知机制**不在本协议范围内。两边可选：

- 人工 ping（最简单）。
- 共享文件 mtime（RL 启动后 watch 响应文件出现）。
- CI / git hook（RL 写完 commit + push，RO 端 CI 跑完 push 响应回来，RL pull）。

---

## 6. 一致性自检（双方都该跑）

RL 写完请求后、读响应前后：

```python
from blb_stage2_rl.baseline_bootstrap import (
    write_baseline_request, read_baseline_response, validate_response_against_request,
)

# 写请求
req_path = write_baseline_request(
    repo_root="...",
    dataset="mrpc",
    stage1_gelu_per_layer=[1, 1, ..., 1, 4, 1, ..., 1],
    stage1_softmax_per_layer=[2, 2, 5, ..., 2],
    num_layers=12,
)

# ... 等 RO 写好响应 ...

# 读响应
result = read_baseline_response(
    repo_root="...",
    dataset="mrpc",
    expected_request_id="20260510-mrpc-bert-base",
)
# 自动校验 schema / request_id / num_layers / results 长度
assert result.ok, f"baseline failed: {result.error}"
```

RO 一侧应在写响应前做 self-check：

1. `results` 长度 == `5 * num_layers - 1`（缺 (block=1, layer=0)，或者返回但被 RL 忽略）。
2. 每条 result 的 `(block, layer)` 配对都唯一。
3. 每条 result 的 `graph_key` 与 `dataset` / `gelu_degree[layer]` /
   `softmax_degree[layer]` 一致（block 3 → `block3_exp_n<softmax_deg[layer]>`，
   block 5 → `block5_n<gelu_deg[layer]>`，block 1/2 → `block{N}_<dataset>`，
   block 4 → `block4`）。
4. `aggregate.valid_block_count + aggregate.invalid_block_count == 5 * num_layers - 1`。
5. `aggregate.total_bits_sum == sum(r.modulus_chain.total_bits for r in results if r.success)`。

---

## 7. 版本演进

- **v1**（当前）：包含 (block, layer) 完整 baseline；degree per-layer 通过
  `stage1_config` 传递。
- **v2**（未来候选）：可加 `rotation_constraints` 字段，让 RL 提前告知 RO 哪些
  rotation 不可启用；目前 v1 的语义是 "RO 自由决定" + RL 在训练时通过
  `apply_rotation_flags_to_cfg` 接收。

schema 字段如果改名 / 改语义 → 必须 bump `schema: "blb_baseline_request_v2"`，
RL 一侧的 `read_baseline_response` 同步加版本分支。

---

## 8. 术语 / 缩写表

| 术语 | 含义 |
|---|---|
| graph_key | RO 一侧的图标识符（剥 `_L<i>` 后），例 `block1_mrpc`。 |
| skeleton | `replan_with_user_actions` 接收的 stage 序列（baseline 选定的 cut points）。 |
| cut point | skeleton 上一个具体 stage 索引（图节点的位置）。 |
| t_baseline | skeleton 上每个 stage 的 baseline SF。 |
| t_new | RL 想替换 t_baseline 的新 SF（每 stage 一个）。 |
| delta_overrides | 节点级别的 SF 覆盖（per-node，非 per-stage）。 |
| modulus_chain | RO 计算出的 (q_head, q_bits, q_tail, total_bits)。 |
| invalid_chain | 模数链非法时的状态描述；与 chain 互斥（合法 → chain 非空 / invalid_chain=null；非法 → chain=null / invalid_chain 非空）。 |
| effective_rotations | RO 选定要插入的 rotation 候选点集合。 |
| BLB | Break-the-Linear-Barrier；本项目的 baseline 论文核心结构（5 个 fused linear blocks）。 |

---

## 9. 与现有代码的对接位置

RL 一侧：

- 写请求 / 读响应：`blb_stage2_rl/baseline_bootstrap.py`
  （`write_baseline_request` / `read_baseline_response`）
- 把响应转换成 `BaselineCostStats`：同上模块的 `baseline_response_to_cost_stats`
- 训练入口检测到 baseline_response 文件存在时**优先**用它，否则 fall back
  到现有的 in-process 计算（`estimate_baseline_cost_stats`）。

RO 一侧：

- 读请求并写响应的实现完全由 RO 团队负责，本仓库不提供（可参考
  `Rescale_optimizer/scripts/replan_what_if.py` 的 JSON 输出风格）。
- 建议在 `Rescale_optimizer/scripts/` 下加 `gen_blb_baseline.py`（对应
  CLI 入口），格式与 `replan_what_if.py` 类似。

---

## 10. 向 RO 一侧的最小实现 checklist

如果 Rescale_optimizer 一侧只想最快接通，可以这样实现 `gen_blb_baseline.py`：

```python
# Rescale_optimizer/scripts/gen_blb_baseline.py（伪代码示意）
import json, sys, os
from rescale_optimizer import load_graph_from_json, ReplanInputs, replan_with_user_actions

def main(req_path, resp_path):
    req = json.load(open(req_path, encoding="utf-8"))
    dataset = req["dataset"]
    L = req["num_layers"]
    gelu = req["stage1_config"]["gelu_degree_per_layer"]
    softmax = req["stage1_config"]["softmax_degree_per_layer"]

    archive_path = f"configs/{dataset}/static_skeletons_{dataset}.json"
    archive = json.load(open(archive_path, encoding="utf-8"))
    by_name = {r["config_name"]: r for r in archive["results"] if r.get("success")}

    results = []
    n_valid = n_invalid = 0
    total_bits = total_fusion = 0
    for layer in range(L):
        for block in (1, 2, 3, 4, 5):
            graph_key = pick_graph_key(block, dataset, gelu[layer], softmax[layer])
            archive_entry = by_name.get(graph_key)
            if archive_entry is None:
                results.append({
                    "config_name": f"{graph_key}_L{layer}",
                    "graph_key": graph_key, "block": block, "layer": layer,
                    "success": False,
                    "error_message": f"baseline archive missing {graph_key}",
                })
                n_invalid += 1
                continue

            graph = load_graph_from_json(f"configs/{dataset}/{graph_key}.json")[0]
            skel = list(archive_entry["skeleton"]) + [graph.M + 1]   # add dummy sink
            t_base = baseline_t_from_archive(archive_entry)
            q_base = baseline_q_from_archive(archive_entry)
            r = replan_with_user_actions(
                graph,
                ReplanInputs(skeleton=skel, t_baseline=t_base, t_new=t_base, delta_overrides=None),
                baseline_q_bits=q_base,
            )

            ok = (r.invalid_chain is None) and r.valid
            results.append({
                "config_name": f"{graph_key}_L{layer}",
                "graph_key": graph_key, "block": block, "layer": layer,
                "success": ok,
                "skeleton": archive_entry["skeleton"],
                "t_baseline": t_base, "q_bits_baseline": q_base,
                "modulus_chain": (
                    None if r.chain is None else {
                        "q_head_bits": r.chain.q_head_bits,
                        "q_bits": list(r.chain.q_bits),
                        "q_tail_bits": r.chain.q_tail_bits,
                        "total_bits": r.chain.total_bits,
                        "R": r.chain.R,
                    }
                ),
                "fusion_count": int(r.fusion_count),
                "invalid_chain": (
                    None if r.invalid_chain is None else {
                        "q_head_bits": r.invalid_chain.q_head_bits,
                        "q_bits": list(r.invalid_chain.q_bits),
                        "q_tail_bits": r.invalid_chain.q_tail_bits,
                    }
                ),
                "graph_path": f"configs/{dataset}/{graph_key}.json",
            })
            if ok:
                n_valid += 1
                total_bits += r.chain.total_bits
                total_fusion += int(r.fusion_count)
            else:
                n_invalid += 1

    resp = {
        "schema": "blb_baseline_response_v1",
        "request_id": req["request_id"],
        "request_path": req_path,
        "dataset": dataset, "model": req["model"], "num_layers": L,
        "ok": (n_invalid == 0),
        "error": None if n_invalid == 0 else f"{n_invalid}/{n_valid + n_invalid} blocks failed",
        "ro_version": "1.0.0",
        "static_skeletons_archive": archive_path,
        "results": results,
        "aggregate": {
            "valid_block_count": n_valid,
            "invalid_block_count": n_invalid,
            "total_bits_sum": int(total_bits),
            "total_fusion_count": int(total_fusion),
        },
    }
    with open(resp_path, "w", encoding="utf-8") as f:
        json.dump(resp, f, ensure_ascii=False, indent=2)


def pick_graph_key(block, dataset, gelu_deg, softmax_deg):
    if block == 1: return f"block1_{dataset}"
    if block == 2: return f"block2_{dataset}"
    if block == 3: return f"block3_exp_n{int(softmax_deg)}"
    if block == 4: return "block4"
    if block == 5: return f"block5_n{int(gelu_deg)}"
    raise ValueError(block)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
```

调用：

```bash
python Rescale_optimizer/scripts/gen_blb_baseline.py \
    handover/baseline_request_mrpc.json \
    handover/baseline_response_mrpc.json
```

把这份 `gen_blb_baseline.py` 放到 `Rescale_optimizer/scripts/` 下、按本协议
读写 JSON，就能与 RL 一侧对接。
