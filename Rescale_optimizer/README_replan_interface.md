# Rescale Optimizer 变量式 Replan 接口

本文档说明新增的 Python in-process 接口。它保留原来的
`scripts/replan_what_if.py --out xxx.json` 文件模式，同时支持 RL 训练循环直接传入
变量并拿到返回值，避免每一步读写 JSON / fork 子进程。

## 1. 推荐入口：`ReplanSession`

```python
from rescale_optimizer import ReplanSession

session = ReplanSession.from_profile(
    profile="mrpc",
    # root 默认是 Rescale_optimizer 仓库根目录；在仓库外调用时显式传入更稳
    root="/var/tmp/root-home/Reinforcement-For-Robustness/Rescale_optimizer",
)

out = session.replan(
    "block3_exp_n2",
    t_new=[27, 34, 34],
    delta_overrides={"ctpt_inv_2n": 16},
)

print(out["valid"])
print(out["fusion_count"])
print(out["result"]["chain"]["total_bits"])
print(out["new_compact_config"])
```

`ReplanSession.from_profile(profile="mrpc")` 会预加载：

- `configs/mrpc/*.json` 里的 graph；
- `configs/mrpc/static_skeletons_mrpc.json` 里的 baseline skeleton / `t_baseline` / baseline q。

之后每次 `session.replan(...)` 只做一次 replan 计算，返回 dict，不写文件。

## 2. 调用形态

### 2.1 显式变量

```python
out = session.replan(
    graph_key="block5_n4",
    t_new=[30, 31, 31, 31, 31],
    delta_overrides={
        "ctpt_gamal": 22,
        "ctpt_wffn1": 22,
        "ctpt_gelu_coeff": 22,
    },
)
```

如果省略 `t_new`，接口会使用 baseline 的 `t_baseline`，适合只评估
`delta_overrides` 的场景：

```python
out = session.replan("block2_mrpc", delta_overrides={"ctpt_wq_wk": 22})
```

### 2.2 Invoker 兼容调用

`ReplanSession` 本身也可当 callable 使用：

```python
out = session("block3_exp_n5", {
    "t_new": [28, 31, 31, 31, 31, 31],
    "delta_overrides": {"ctpt_inv_2n": 15},
})
```

兼容旧的 bare-dict 形式：

```python
out = session("block1_mrpc", {"ctpt_ffn2": 22})
```

这个 bare dict 会被解释成 `delta_overrides`，`t_new` 自动取 baseline。

## 3. 单次函数：`replan_from_variables`

不需要预加载时，可以直接单次调用：

```python
from rescale_optimizer import replan_from_variables

out = replan_from_variables(
    config_path="configs/mrpc/block3_exp_n2.json",
    baseline_archive="configs/mrpc/static_skeletons_mrpc.json",
    config_name="block3_exp_n2",
    t_new=[27, 34, 34],
    delta_overrides={"ctpt_inv_2n": 16},
)
```

也可以不传 `baseline_archive`，而是直接传入变量：

```python
out = replan_from_variables(
    config_path="configs/mrpc/block3_exp_n2.json",
    skeleton=[0, 2, 3, 4],
    t_baseline=[27, 34, 34],
    q_bits_baseline=[52, 34],
    t_new=[27, 34, 34],
)
```

## 4. 返回值结构

返回 dict 与旧 `replan_what_if.py --out` 的 JSON 结构兼容，并新增 `graph_key`：

```json
{
  "config_name": "block3_exp_n2",
  "graph_key": "block3_exp_n2",
  "valid": true,
  "fusion_count": 0,
  "baseline": {
    "skeleton": [0, 2, 3, 4],
    "t_baseline": [27, 34, 34],
    "q_bits_baseline": [52, 34]
  },
  "t_new": [27, 34, 34],
  "delta_overrides": {"ctpt_inv_2n": 16},
  "result": {
    "valid": true,
    "message": "replan OK after 0 fusion(s). R: 2 -> 2.",
    "fusion_count": 0,
    "skeleton": [0, 2, 3, 4],
    "q_initial": [52, 34],
    "q_final": [52, 34],
    "t_final": [27, 34, 34],
    "delta_q_vs_baseline": [0, 0],
    "applied_delta_overrides": {"ctpt_inv_2n": 16},
    "chain": {
      "q_head_bits": 60,
      "q_bits": [52, 34],
      "q_tail_bits": 60,
      "total_bits": 206,
      "R": 2
    },
    "invalid_chain": null
  },
  "new_compact_config": {
    "config_name": "block3_exp_n2",
    "success": true,
    "skeleton": [0, 2, 3, 4],
    "cut_point_sf": [],
    "propagation_deltas": [],
    "modulus_chain": {},
    "effective_rotations": []
  }
}
```

如果只想要底层 `ReplanResult` dataclass：

```python
result = session.replan("block3_exp_n2", return_dict=False)
```

## 5. Stage-1 degree 到 graph key 的映射

协议中 Stage-1 决定每层 Softmax/GELU degree。新增 helper 可把它转换成
Rescale_optimizer 的 graph key：

```python
from rescale_optimizer import graph_key_for_stage1, iter_stage2_graph_targets

stage1 = {
    "gelu_degree_per_layer":    [1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1],
    "softmax_degree_per_layer": [2, 2, 5, 5, 5, 2, 5, 2, 5, 5, 6, 2],
}

graph_key = graph_key_for_stage1(
    dataset="mrpc",
    block=3,
    layer=2,
    stage1_config=stage1,
)
# graph_key == "block3_exp_n5"

targets = iter_stage2_graph_targets(
    dataset="mrpc",
    num_layers=12,
    stage1_config=stage1,
)
```

如果直接使用 `Model_analysis/configs/approx_per_dataset.json` 中某个
`<dataset>/<stage>` 小节，也支持短字段名：

```python
stage1 = {
    "gelu":    [1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1],
    "softmax": [2, 2, 5, 5, 5, 2, 5, 2, 5, 5, 6, 2],
}
```

映射规则：

- block1: `block1_<dataset>`，但按协议默认跳过 `(block=1, layer=0)`；
- block2: `block2_<dataset>`；
- block3: `block3_exp_n<softmax_degree_per_layer[layer]>`；
- block4: `block4`；
- block5: `block5_n<gelu_degree_per_layer[layer]>`。

## 6. 与两个协议文档的关系

- `blb_baseline_handover_protocol(1).md` 的 JSON baseline 握手仍然可以在外层实现；
  本接口提供每个 `(block, layer)` 的实际 replan 计算和返回 dict。
- `blb_rl_to_rescale_optimizer_mapping(1).md` 中的 `t_new` / `delta_overrides`
  就是本接口的输入变量。RL 侧根据 action slot 解码出这些变量后，直接调用
  `session.replan(...)` 即可。

## 7. 测试

```bash
cd /var/tmp/root-home/Reinforcement-For-Robustness/Rescale_optimizer
python scripts/test_replan_interface.py
```

该测试覆盖：

- profile session 预加载；
- baseline `t_new` 默认路径；
- rich payload 调用；
- 单次 `replan_from_variables(...)`；
- Stage-1 degree 到 graph key 的映射。
