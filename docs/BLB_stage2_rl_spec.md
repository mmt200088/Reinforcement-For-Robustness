# 加强版 Stage 2 强化学习设计 spec —— 给另一个 Agent 执行

> 这份文档是给一个**没有任何上下文**的 Claude Opus 4.7 agent 读的。读完它你应当
> 能完全理解：(1) 项目背景；(2) 我们要 RL 优化的对象；(3) 动作 / 状态 / 奖励 /
> 算法 / 训练循环 / 文件布局；(4) 怎样落地实现而不破坏旧代码。所有依赖的代码
> 入口（类名、函数名、文件路径、字段名）都已在文中明确写出，**不要凭空发明**。
> 如有疑问，请直接读 [`function_handler.py`](../function_handler.py) /
> [`blb_rl_bridge.py`](../blb_rl_bridge.py) /
> [`rescale_optimizer_bridge.py`](../rescale_optimizer_bridge.py)，那才是真相。

---

## 0. 项目是什么

仓库 `d:\Desktop\隐私计算LLM\Local_program` 是一个 BERT 上的**密文隐私推理（PPTI）
明文态模拟器**。原系统在 CKKS 同态加密 + 多方安全计算（MPC）下做 transformer
推理；本仓库不真的跑 CKKS / MPC，而是在普通 PyTorch BERT 前向中**注入与 CKKS / MPC
等价的高斯噪声 + 截断**，用以快速搜索"哪些位置该放 rescale / rotation / 用什么
scaling factor / 截多少 bit" 这类工程参数。

我们已经在 5 个 BLB block 上铺好了所有候选噪声点：fresh / encode / rescale /
rotation / 末尾 truncation。本文要做的事是**用强化学习自动选这些噪声点的
scaling factor 与 truncation k**，目标是最小化部署在真实 CKKS+MPC 下的开销
（模数链 total_bits、fusion_count、truncation 后保留的 bit 数 k）的同时，
不破坏模型精度与稳定性。

---

## 1. 名词术语（务必先看完）

### 1.1 BLB

"Breaking the Layer Barrier"。把一个 transformer encoder layer 跨层切成 5 个
block：

| Block | 范围（粗略）                                                           | LayerNorm 替身                                |
| ----- | ---------------------------------------------------------------------- | --------------------------------------------- |
| 1     | 上一层 GELU 输出 → 本层 Wffn2 → post-FFN LN head（rsqrt 之前）         | `output.LayerNorm` 的 head 部分               |
| 2     | post-FFN LN tail → Wq/Wk/Wv → BSGS mask × 2 → Q·K^T → merge mask       | `output.LayerNorm` tail + 下一层 attention.self |
| 3     | softmax 多项式 (1 + x/2^n)^(2^n)                                       | `attention.self.approximation_exponential`    |
| 4     | softmax 输出 + V → softmax×V → merge mask → Wo → post-attn LN head     | `attention.output.LayerNorm` head             |
| 5     | post-attn LN tail → Wffn1 → GELU 多项式                                | `attention.output.LayerNorm` tail + GELU      |

首层的 Block 1 不启用 SF/Gaussian/rotation 噪声，也不进入 RO replan；但其
LayerNorm variance → rsqrt 边界存在，因此 Block 1 truncation K 与其它层一样
由 RL 选择并真实执行。Block 2 末尾 Q·K^T 也照常 truncation。

### 1.2 CKKS 噪声四类

- **fresh**：明文加密 / MPC→HE 转换时引入的"新鲜"密文噪声。每次 fresh
  操作都让消息乘上一个 `2^sf`（sf = scaling factor），表中 `fresh` 列是对应的
  噪声方差 σ²。
- **encode**：服务端把明文常数（W、γ、1/D、ones-mask、1/2^n、多项式系数）encode 成
  plaintext 时引入的噪声。同样对应一个 sf。
- **rescale**：CKKS 不能无限累乘；每隔几次乘法就要 rescale 把 scaling factor
  压回去。rescale 也带噪声，对应一个 sf。**rescale 是否真在某个候选点发生**
  由 `Rescale_optimizer` 算法决定，不是 RL 直接选的。
- **rotation**：KS / galois automorphism 引入的 keyswitch 噪声。**绑定**到它前面
  的 fresh / rescale —— SF 直接复用绑定源的 SF；噪声方差表里 `rotation` 列与
  `rescale` 列等同。rotation 是否触发也由 `Rescale_optimizer` 决定（"effective
  rotation"）。

### 1.3 "当前 scaling factor"（CSF）—— 必须搞懂

每条密文有一个隐式 CSF，记录当前累积的缩放：
- 一次 ctct (ct·ct) 乘法：CSF ← CSF_a + CSF_b
- 一次 ctpt (ct·pt) 乘法：CSF ← CSF_ct + sf_encode_or_fresh
- 一次 rescale：CSF ← rescale_sf （直接拉到 rescale 选定的 sf 值）
- truncation：与 CSF 无关，只截掉小数 bit（模拟 MPC 端做完返回 HE 前的精度损失）

**RL 的本质：通过选择 fresh / encode / rescale / 各 SF + truncation k，操纵每个
block 内 CSF 的演化，并由 `Rescale_optimizer` 决定哪些候选 rescale 真的触发，
最终给出模数链开销与有效 rotation 列表。**

### 1.4 truncation k

每个 block 末尾的输出做 `floor(x · 2^k) / 2^k`（binary 模式）。k 越小 → 损失
越大、开销越小；k 越大 → 精度越好、开销越大。

---

## 2. 仓库地图（已实现）

```
Local_program/
├── function_handler.py            ★ BLB 噪声内核
│   - NoisePoint dataclass: (distribution, scaling_factor, N)
│   - NOISE_VARIANCE_TABLE_BY_N[N][sf]["encoding/fresh/rescale/rotation"]
│   - get_input_noise_variance_by_N(...)
│   - _make_rotation_point(source) -> rotation NoisePoint，继承 SF/N
│   - _apply_truncation(x, k, mode="binary")
│   - Block{1..5}NoiseConfig dataclass（每块的所有 noise 字段，含 rotation_after_*: bool）
│   - make_block{1..5}_default_config(...) factory
│   - NoisyBlock1LayerNorm / NoisyBlock4LayerNorm（替换 LN）
│   - _make_block{1..5}_*_forward / *_hook（注入噪声的闭包）
│   - ReversibleLayerHandler:
│       replace_layer_block{1..5}_noise(layer_indices, layer_name, cfg)
│       restore_layer_block{1..5}_noise(...)
│       replace_blb_first_input_noise(scaling_factor, N, ...)
│       restore_blb_first_input_noise(...)
│       _check_blb_legacy_conflict(...)        # BLB 与 legacy 互斥
│       restore_all()
│
├── blb_rl_bridge.py               ★ RL 动作 → BLB cfg → install/restore
│   - BLBNoiseRLBridge.apply(first_input_sf, block{1..5}_cfgs={layer:cfg})
│   - BLBNoiseRLBridge.clear()
│   - BLBNoiseRLBridge.installed_layers()
│   - Block{1..5}ActionSpec dataclass（per-layer 一组动作字段）
│   - build_block{1..5}_cfg_from_action(action, N=...) -> Block{}NoiseConfig
│   - aggregate_truncation_signals(cfg_dicts) -> TruncationRewardSignals
│   - aggregate_rotation_signals(cfg_dicts)   -> RotationRewardSignals
│   - BLB_DEFAULT_ALLOWED_SFS_FRESH / ENCODE / RESCALE
│   - discrete_action_to_sf / discrete_action_to_optional_sf
│
├── rescale_optimizer_bridge.py    ★ RL 奖励侧黑盒桥接
│   - RescaleOptimizerOutput(fusion_count, total_bits, invalid_chain, valid, raw)
│   - RescaleOptimizerBridge.evaluate(config_name, block_name, cfg) -> Output
│   - RescaleOptimizerBridge.evaluate_blocks(requests) -> {config_name: Output}
│   - aggregate_optimizer_signals(outputs) -> OptimizerRewardSignals
│   - apply_rotation_flags_to_cfg(cfg, rotation_flag_names: Iterable[str])
│   - InProcessInvoker（训练固定使用）/ SubprocessInvoker / CallableInvoker / StubInvoker（仅桥接层调试）
│   - default_block{1..5}_cfg_to_delta(cfg) -> {node_name: delta_int_or_'x2'}
│
├── Rescale_optimizer/             ☆ 外部子项目（已经拉下来，配 configs/mrpc 等）
│   ├── rescale_optimizer/                                 Python 包（可直接 import）
│   │   ├── replan.py    ReplanInputs / replan_with_user_actions
│   │   └── ...
│   ├── configs/<profile>/block*_<profile>.json            输入图（多 block）
│   ├── configs/<profile>/static_skeletons_<profile>.json  baseline 归档（含 t/skeleton/q）
│   ├── replan_configs/<profile>/replan_actions_*.json     可选：actions JSON 模板
│   └── scripts/replan_what_if.py                          what-if replan CLI（subprocess 用）
│   见 docs/README_configs.md 和 docs/README_rotations.md。
│
├── layer_importance_evaluator.py  ☆ 旧版 stage 2 RL（PPO over legacy noise，不要碰）
│   - apply_input_noise_configuration / apply_wq_noise_configuration / ...
│   - 用 INPUT_NOISE_VARIANCE_TABLE 单 N 表
│   - 与 BLB 已经强制互斥（function_handler 里的 _check_blb_legacy_conflict）
│
└── docs/                          ← 你将把本 spec 也放在这里
```

---

## 3. RL 任务定义

### 3.1 优化目标（一句话）

在 BLB 5 个 block 的所有候选噪声点上为每层选择
**(scaling_factors, truncation_k, rotation_flags)**，
使得：

1. **精度约束**：经过噪声前向后，模型在评估子集上的精度 ≥ 阈值；
2. **稳定性约束**：精度方差 / loss 方差 ≤ 阈值；
3. 在前两条都满足的前提下，最小化 **部署侧的 CKKS + MPC 总开销**：
   `cost = w_t · (-Δtotal_bits) + w_f · fusion_count + w_k · (-Δk)`
   （Δ 是相对"全部用最大 SF"的基线下降量）

精度 / 稳定性是**硬约束式 priority** —— 不满足时本次动作直接被否决，没有 cost
奖励；只在两条硬约束都过线后才比较 cost。

### 3.2 一回合（episode）流程伪代码

```python
# 1. agent emits action（动作向量见 §4）
action = agent.sample(state)

# 2. action → per-layer Block{1..5}NoiseConfig（含 rotation flags 暂时全 False）
cfgs = build_all_cfgs(action, max_sfs)        # see §4.5

# 3. 调 Rescale_optimizer，拿到 fusion_count / total_bits / invalid_chain
#    + effective_rotations + 对应 rescale 模式（哪些 rescale slot 真触发）
opt_outputs = rescale.evaluate_blocks({
    f"block{b}_<profile>": (f"block{b}", cfgs[b][i])
    for b in (1,2,3,4,5) for i in selected_layers
})
opt_signals = aggregate_optimizer_signals(opt_outputs)

# 4. 任一 invalid_chain=True → 动作非法，跳过本步（见 §6.5）
if opt_signals.any_invalid:
    return invalid_action_handling(...)

# 5. 把 effective rotations 反写到 cfg 上
for cfg, opt_out in pair(cfgs, opt_outputs):
    rotation_flag_names = map_optimizer_rotations_to_blb_flags(
        opt_out.raw["new_compact_config"]["effective_rotations"]
    )
    apply_rotation_flags_to_cfg(cfg, rotation_flag_names)

# 6. 把所有 BLB 噪声装到模型
blb_bridge.apply(
    first_input_sf=action.first_input_sf, first_input_N=16384,
    block1_cfgs=cfgs[1], block2_cfgs=cfgs[2],
    block3_cfgs=cfgs[3], block4_cfgs=cfgs[4],
    block5_cfgs=cfgs[5],
)

# 7. 在评估子集上跑 forward；得到 acc / 稳定性
metrics = eval_on_probe_subset(model, ids, mask, labels)

# 8. 还原模型（必须）
blb_bridge.clear()

# 9. 计算 reward（见 §6）
reward = compute_reward(metrics, opt_signals, action, max_action_baseline)
```

---

## 4. 动作空间

### 4.1 类型 → 挡位数

| 噪声口径   | 用途                                     | 挡位数 | 说明                  |
| ---------- | ---------------------------------------- | ------ | --------------------- |
| `W`        | weight encode (Wq/Wk/Wv/Wo/Wffn1/Wffn2)  | 5      | sf ∈ {max-8, max-6, max-4, max-2, max} |
| `M` / `S`  | mask / 标量 encode (γ, 1/D, ones, 1/2^n) | 3      | sf ∈ {max-4, max-2, max} |
| `R`        | rescale                                   | 4      | sf ∈ {max-6, max-4, max-2, max} |
| `F`        | fresh                                     | 5      | sf ∈ {max-8, max-6, max-4, max-2, max} |
| `K`        | 末尾 truncation 保留位数                  | 8      | `K_LEVELS=(8, 9, 11, 13, 10, 12, 6, 7)` |
| `B`        | rotation 是否启用                         | 2      | bool ∈ {0, 1}         |

`max` 来自 `Rescale_optimizer/configs/<profile>/static_skeletons_<profile>.json`
里每个 block 每个 cut_point 的 baseline `sf` / `sf_pre`（详见
[`docs/README_configs.md`](README_configs.md) 中的 ❷ 输出）。**不要写死**：
读 JSON 拿 max，再按 `step=2` 反推 5/3/4 个挡位。

`B` 不是 RL 选的（rotation 启用与否由 Rescale_optimizer 决定），但保留在
`Block{}NoiseConfig` 里供 `apply_rotation_flags_to_cfg` 写回（详见 §5.4）。

`K_LEVELS` 的顺序是 checkpoint/action-index 契约，不是数值排序：历史
indices `0..5` 保持不变，index `3` 仍是 baseline `K=13`，新 indices
`6`/`7` 分别为 `K=6`/`K=7`。canonical layerwise policy 每层仍有六个
categorical slots（一个 Block4 fusion + 五个 K），但每个 K slot 现在有八类。
成本与 reward 按 `13-6` 归一化；旧六类 checkpoint 必须 fresh run。K 与
fusion-map option 分离且旧 indices 未改变，所以本次扩展不要求重建 fusion
maps。

### 4.2 每层每 block 的动作维度（mandatory + optional）

每个 block 的精确字段对照 [`function_handler.py`](../function_handler.py) 中的
`Block{N}NoiseConfig` dataclass。这里给出 RL 该出多少个 categorical action：

| Block | F (fresh) | W (weight) | M / S | R (rescale) | K | 总 SF dim |
| ----- | --------- | ---------- | ----- | ----------- | - | --------- |
| 1     | 1 (gelu_out) | 1 (wffn2) | 2 (mean,var的1/D) | 4 (wffn2/mean/sq/var) | 1 | 8 + 1K |
| 2     | 2 (inv_std,xc) | 3 (wq,wk,wv) | 1+5=6 (γ + 5 mask) | 11 (各 *_rescale) | 1 | 22 + 1K |
| 3 (deg=4) | 1 (x_softmax) | 0 | 1 (1/2^n) | 1 + degree = 5 | 1 | 7 + 1K |
| 4     | 2 (sm_out,V) | 1 (wo) | 3+2=5 (3 mask + 2 LN inv_d) | 8 (各 *_rescale) | 1 | 16 + 1K |
| 5 (deg=4) | 2 (inv_std,xc) | 1 (wffn1) | 1+1=2 (γ + gelu_coeff) | 3 + (deg-1) + deg = 10 | 1 | 15 + 1K |
| first-input | 1 | 0 | 0 | 0 | 0 | 1 |

每层 **94 个** 离散动作（softmax_deg=4, gelu_deg=4 时）。L=12 层 → ≈ **1129 个**
离散动作。

> 注：optional rescale 在我们的 cfg 里也是 `Optional[NoisePoint]`，传 None 表示"不
> 在该候选点装 rescale"。当前动作语义中，R 槽 index 0 表示 off / None；
> 其余 index 表示真实 SF。这样 static_skeletons baseline 中没有 `sf_post/drop`
> 的 rescale 点不会被明文噪声模拟误安装。

### 4.3 动作空间的形式化

定义全局动作向量 `a = (a_layer_0, a_layer_1, ..., a_layer_{L-1}, a_first_input)`，
每个 `a_layer_i` 是一个嵌套 dict，结构和 `Block{N}ActionSpec` 完全对应。

工程上推荐 **MultiDiscrete** 表达：

```python
# 在 RL 框架（gymnasium 风格）里：
action_space = MultiDiscrete([
    # Layer 0 Block 1 (8 SF + 1 K)：
    5, 5, 3, 3, 4, 4, 4, 4, 4,
    # Layer 0 Block 2 (22 SF + 1 K) ...
    ...
    # ...
])
total_dim = L * 94 + 1   # = 1129 for L=12
```

存成一个一维数组，每个分量是该位置的挡位 index。policy 网络输出
`[total_dim]` 个独立 Categorical。

### 4.4 max SF 的获取（不要写死）

```python
# 伪代码：每个 (block, profile) 一份 max_sf 表
import json
def load_max_sfs(skeleton_json_path):
    """从 static_skeletons_<profile>.json 读出每个 cut_point 的 max sf。"""
    with open(skeleton_json_path) as f:
        archive = json.load(f)
    out = {}  # config_name -> {cut_point_name: max_sf}
    for r in archive["results"]:
        if not r.get("success"):
            continue
        per_node = {}
        for cp in r["cut_point_sf"]:
            per_node[cp["name"]] = int(cp.get("sf_pre", cp.get("sf", 0)))
        out[r["config_name"]] = per_node
    return out
```

之后 RL 用 `max_sf - 2*(num_levels-1-idx)` 反查具体 SF。`num_levels` 按 §4.1 表
取 5 / 3 / 4。

### 4.5 action → cfg 翻译

```python
def build_all_cfgs(action_vec, max_sfs, gelu_degrees, attn_degrees, layer_count):
    """action_vec: np.ndarray[int] of shape [total_dim]
       max_sfs: {(block_name, node_name): int}（来自 §4.4）
       returns {1: {layer:cfg1}, 2: {layer:cfg2}, ..., 5: {layer:cfg5}, "first_input_sf": int}
    """
    pos = 0
    cfgs = {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}
    for li in range(layer_count):
        # Block 1
        a = Block1ActionSpec(
            gelu_out_sf=    sf_from(action_vec[pos+0], max=max_sfs[("block1","gelu_out")],   levels=5);  pos+=1,
            wffn2_sf=       sf_from(action_vec[pos+0], max=max_sfs[("block1","ctpt_ffn2")],  levels=5);  pos+=1,
            mean_inv_d_sf=  sf_from(action_vec[pos+0], max=max_sfs[("block1","ctpt_inv_d_1")], levels=3); pos+=1,
            var_inv_d_sf=   sf_from(action_vec[pos+0], max=max_sfs[("block1","ctpt_inv_d_2")], levels=3); pos+=1,
            wffn2_rescale_sf=  sf_from(action_vec[pos+0], levels=4); pos+=1,
            mean_rescale_sf=   sf_from(action_vec[pos+0], levels=4); pos+=1,
            square_rescale_sf= sf_from(action_vec[pos+0], levels=4); pos+=1,
            var_rescale_sf=    sf_from(action_vec[pos+0], levels=4); pos+=1,
            output_truncation_k=K_LEVELS[action_vec[pos+0]];          pos+=1,
        )
        cfgs[1][li] = build_block1_cfg_from_action(a, N=8192)
        # ... blocks 2..5 同理 ...
    cfgs["first_input_sf"] = sf_from(action_vec[pos], levels=5)
    return cfgs

K_LEVELS = (8, 9, 11, 13, 10, 12, 6, 7)  # action_idx ∈ {0,...,7}; idx3 = baseline K13
def sf_from(idx, max, levels): return max - 2 * (levels - 1 - int(idx))
```

> ⚠️ 上面 `max_sfs[(block_name, node_name)]` 的具体节点名要对照
> [`docs/README_configs.md`](README_configs.md) 里 `cut_point_sf[i].name`（例如
> block1 的 `ctpt_ffn2`/`ctct_ext_square`/`ctpt_inv_d_1`、block2 的
> `ctpt_gama1`/`ctpt_wq_wk`、block5 的 `ctpt_wffn1`/`ctct_gelu_x2` 等；
> 注意 BLB cfg 里独立的 `wq` / `wk` 在 graph 里被融合为一个 `ctpt_wq_wk`）。
> 我们已经在 [`rescale_optimizer_bridge.py`](../rescale_optimizer_bridge.py) 的
> `default_block{1..5}_cfg_to_delta` 里写了一份默认命名对照，可以反向用。

---

## 5. 状态空间与外部接口

### 5.1 state 设计

最小 state（推荐起步用，足够 PPO 收敛）：

```python
state = np.concatenate([
    # 静态 context（不随 episode 变化）
    [softmax_degree, gelu_degree, num_layers, profile_id_onehot...],
    # 当前样本统计（在每个 episode 开始重新采样的探针子集）
    [probe_loss_baseline, probe_acc_baseline],
    # 上一回合的全局 cost & invalid 比例（用于学习 invalid 边界）
    [last_total_bits_norm, last_fusion_count, last_invalid_rate],
])
```

可选升级（autoregressive policy）：每层各 block 独立决策时，state 还附上
"前面已选的 SF 中位数 / 上一层选了什么"。但起步阶段一次性出全向量已够。

### 5.2 外部接口（必读）

#### 5.2.1 安装 / 还原噪声

[`BLBNoiseRLBridge.apply / clear`](../blb_rl_bridge.py)：

```python
bridge = BLBNoiseRLBridge(handler, layers_attribute="model.bert.encoder.layer")
bridge.apply(
    first_input_sf=action.first_input_sf, first_input_N=16384,
    block1_cfgs={i: cfg1_for_layer_i for i in selected_layers},
    block2_cfgs=..., block3_cfgs=..., block4_cfgs=..., block5_cfgs=...,
)
logits = model(input_ids=..., attention_mask=...).logits
bridge.clear()
```

`clear()` 会按 5 → 4 → 3 → 2 → 1 → first_input 反向还原。**每个 episode 必须配对
`apply` / `clear`**，否则下一个 episode 的噪声会叠加。

#### 5.2.2 调 Rescale_optimizer

[`RescaleOptimizerBridge.evaluate_blocks`](../rescale_optimizer_bridge.py)。**推荐用
`InProcessInvoker`**（直接 `import rescale_optimizer` 调
`replan_with_user_actions`，预加载图 + baseline，单步 ms 级；不用 fork）：

```python
from rescale_optimizer_bridge import (
    InProcessInvoker, RescaleOptimizerBridge, aggregate_optimizer_signals,
)

inv = InProcessInvoker.from_profile(
    rescale_optimizer_root="Rescale_optimizer",
    profile="mrpc",
    # configs_dir 默认 = <root>/configs/mrpc
    # baseline_archive 默认 = <configs_dir>/static_skeletons_mrpc.json
)
rescale = RescaleOptimizerBridge(invoker=inv)
outputs = rescale.evaluate_blocks(
    requests={
        "block1_mrpc":   ("block1", cfg1_layer0),
        "block2_mrpc":   ("block2", cfg2_layer0),
        "block3_exp_n4": ("block3", cfg3_layer0),
        "block4":        ("block4", cfg4_layer0),
        "block5_n4":     ("block5", cfg5_layer0),
    },
    # 可选：t_new_per_config={"block1_mrpc": [30, 34, 34], ...}
    # 不传 ⇒ 走 baseline t（等价于"只改 delta_overrides"）
)
signals = aggregate_optimizer_signals(outputs)  # OptimizerRewardSignals
```

桥接层调试时也可以单独走 `SubprocessInvoker`（fork `python scripts/replan_what_if.py`，
开销几百 ms，适合 debug 隔离）。BLB Stage-2 RL 训练路径不使用 subprocess：

```python
from rescale_optimizer_bridge import SubprocessInvoker

inv = SubprocessInvoker(
    rescale_optimizer_root="Rescale_optimizer",
    configs={
        "block1_mrpc": "Rescale_optimizer/configs/mrpc/block1_mrpc.json",
        "block2_mrpc": "Rescale_optimizer/configs/mrpc/block2_mrpc.json",
        "block3_exp_n4": "Rescale_optimizer/configs/mrpc/block3_exp_n4.json",
        "block4": "Rescale_optimizer/configs/mrpc/block4.json",
        "block5_n4": "Rescale_optimizer/configs/mrpc/block5_n4.json",
    },
    baseline_archive="Rescale_optimizer/configs/mrpc/static_skeletons_mrpc.json",
    cli_script="scripts/replan_what_if.py",   # 默认即此值
)
```

正式 BLB Stage-2 RL 训练必须使用真实 `Rescale_optimizer` in-process 路径；如果没有装好，
训练会直接报错停止。`StubInvoker` / `HeuristicStubInvoker` 只保留给桥接层单元测试。

**Action JSON 形态**（invoker 内部 payload）：

```json
{
  "t_new":             [30, 34, 34],
  "delta_overrides":   {"ctpt_ffn2": 20, "ctpt_inv_d_1": 20,
                        "ctct_ext_square": "x2", "ctpt_inv_d_2": 20}
}
```

`t_new` 长度必须 == baseline skeleton 的 R+1（R = rescale 次数）；`delta_overrides`
的 key 是 graph 里 multiplication 节点名（**不是 BLB 命名！** 实际节点名见
`configs/<profile>/block*.json` 的 `stages[].cut_point.name`）。bridge 内置的
`default_block{1..5}_cfg_to_delta` 已对齐 mrpc 节点名（见
[`rescale_optimizer_bridge.py`](../rescale_optimizer_bridge.py)）。

#### 5.2.3 effective rotations 反写

```python
# Rescale_optimizer 输出会有 raw["new_compact_config"]["effective_rotations"]
# 见 docs/README_rotations.md。每条记录有 name (graph 节点名) + after_cut_point 等
for config_name, out in outputs.items():
    eff = out.raw.get("new_compact_config", {}).get("effective_rotations", [])
    blb_flag_names = optimizer_rot_to_blb_flag(config_name, eff)  # ← 你自己写
    apply_rotation_flags_to_cfg(cfgs[block_of(config_name)][layer_of(config_name)], blb_flag_names)
```

`optimizer_rot_to_blb_flag` 是个 per-block 命名表，比如
`{"rot_gs_wffn1": "rotation_after_wffn1_rescale", ...}`。这表 **不要硬编码**到 RL
代码里 —— 放进一个 JSON / Python dict 配置文件。

#### 5.2.4 互斥保证

[`ReversibleLayerHandler._check_blb_legacy_conflict`](../function_handler.py)
会在 `apply` 时自动校验。当抛 `RuntimeError("BLB 噪声与 legacy 噪声互斥...")` 时
意味着旧版 stage 2 残留 ── 你的 RL **不应该**触发这个错误，所以训练前必须确保
模型上没有 legacy 噪声。

```python
# 训练循环开始前一次性清干净
handler.restore_layer_input_noise(layer_indices=range(L))
handler.restore_layer_query_noise(...)  # 等
# ... 其它 legacy restore
# 也可以直接 handler.restore_all()（不过那会 deepcopy，慢）
```

### 5.3 评估子集（probe）

参考旧版 stage 2 在 [`layer_importance_evaluator.py`](../layer_importance_evaluator.py)
的 `stage2_probe_size`（默认 256）。每个 episode 重采一个 probe，跑 forward 拿
acc + loss。**不要在整个 dev set 上跑** —— 那会让一回合数十秒，RL 没法收敛。

### 5.4 rotation flag 反写时机

`apply_rotation_flags_to_cfg(cfg, names)` 把 cfg 上所有 `rotation_after_*` 字段中
**在 names 里出现的置 True，其余置 False**。所以调用前 cfg 里 rotation flag 全为
False 也无所谓。

---

## 6. 奖励函数（核心）

### 6.1 三层优先级

```python
def compute_reward(metrics, opt_signals, action, baseline):
    # 优先级 1：精度约束
    if metrics.acc < ACC_THRESHOLD:
        return -PRIORITY1_PENALTY + (metrics.acc - ACC_THRESHOLD) * PRIORITY1_SCALE
        # 强惩罚 + 精度差距 dense reward，引导 agent 越做越接近阈值

    # 优先级 2：稳定性约束
    if metrics.std > STABILITY_THRESHOLD:
        return -PRIORITY2_PENALTY + (STABILITY_THRESHOLD - metrics.std) * PRIORITY2_SCALE

    # 优先级 3：cost 优化
    return _cost_reward(opt_signals, action, baseline)
```

`PRIORITY1_PENALTY > PRIORITY2_PENALTY > 0 >> _cost_reward 量级`，比如 100 / 50 /
[-3, +3]。具体常数让训练侧 sweep。

### 6.2 cost reward 公式

```python
def _cost_reward(opt_signals, action, baseline):
    # 1. invalid_chain → 直接判死
    if opt_signals.any_invalid:
        return -INVALID_PENALTY        # 见 §6.5

    # 2. Δtotal_bits（vs 全 max-action 的 baseline）
    bits_drop = baseline.total_bits_sum - opt_signals.total_bits_sum
    r_bits = bits_drop * (S / 30.0)

    # 3. fusion_count（越少越好）
    r_fusion = -opt_signals.total_fusion_count * S

    # 4. Δk（vs 全 max-k=13 的 baseline）
    avg_k = action_avg_truncation_k(action)
    k_drop = (BASELINE_AVG_K - avg_k)   # = 13 - avg_k
    r_k = k_drop * S

    return r_bits + r_fusion + r_k
```

`S` 是基础奖励量级（例如 1.0），用户明说：
- `r_fusion` 与 `r_k` 量级一致（同 `S`）；
- `r_bits` 是它们的 `1/30`。

### 6.3 baseline 怎么取

baseline = 把所有 SF 都拉到对应的 max（最高挡位）、所有 truncation k = 13
（最高挡位）。在 RL 训练开始前**跑一次** baseline forward + Rescale_optimizer，
存下 `baseline.total_bits_sum`、`baseline.total_fusion_count`、
`BASELINE_AVG_K = 13`，整个训练复用。

### 6.4 reward 量级标度建议

用户原话：**"`r_fusion` 与 `r_k` 量级一致，`r_bits` 是它们的 1/30"** —— 这里说的是
**加权后的实际 reward 项数值**，不是权重的比。需要根据 baseline 跑出的典型 bits_drop /
fusion_count / k_drop 范围反推权重。**起步建议**（要在第一次 baseline 后校准）：

```python
S = 1.0    # 单位奖励量级

# 校准步骤（训练第一次跑前做）：
#   1. 跑 baseline（全 max-action）→ 拿到 baseline.total_bits_sum
#   2. 跑若干 random action → 估计典型 bits_drop_mean / fusion_count_mean / k_drop_mean
#   3. 反推权重：
#        w_fusion = S
#        w_k      = S
#        w_bits   = (S / 30) / max(1, bits_drop_mean)   ← 让 r_bits ≈ S/30
#      （或者：w_bits = (typical fusion_count) / (30 * typical bits_drop) * S）

# 三层硬约束惩罚（与 cost 量级隔离 1-2 个数量级）：
PRIORITY1_PENALTY = 100.0   # 精度违反基础罚
PRIORITY1_SCALE   = 200.0   # 精度差距 dense 引导
PRIORITY2_PENALTY = 50.0
PRIORITY2_SCALE   = 100.0
INVALID_PENALTY   = 30.0
```

精度 / 稳定性的最大正值远小于负惩罚，确保 RL 永远先把硬约束做满再去抠 cost。

### 6.5 cost reward 完整公式（用 §6.4 的权重）

```python
def _cost_reward(opt_signals, action, baseline, w_bits, w_fusion, w_k):
    if opt_signals.any_invalid:
        return -INVALID_PENALTY
    bits_drop = baseline.total_bits_sum - opt_signals.total_bits_sum
    avg_k     = action_avg_truncation_k(action)
    k_drop    = baseline.avg_k - avg_k          # baseline.avg_k = 13
    return (w_bits   * bits_drop
            + w_fusion * (-opt_signals.total_fusion_count)
            + w_k      * k_drop)
```

### 6.6 invalid_chain 怎么处理

用户说 "invalid_chain=True 那么本次动作无效，不可选择"。两条路径：

**A. 简单（推荐先用）**：给 `-INVALID_PENALTY` reward 让 agent 学着避开。

**B. 严格 action masking**：在 sample 时先用 Rescale_optimizer 跑一遍，invalid
就 reject + resample。代价：每步多花一次 optimizer 调用。

**起步阶段用 A**，等 invalid 占比稳定后再考虑 B。

---

## 7. 算法选型

### 7.1 PPO + MultiDiscrete

强烈推荐 PPO（不要 DQN —— 动作维度太高）。框架可选 `stable-baselines3`、
`ray.rllib`、或基于 PyTorch 自己撸（参考旧版 stage 2 的 PPO 实现，但不要复用
那份代码 —— 它用 ndarray-based 单 N 表，不兼容 BLB 多 N 表）。

### 7.2 Policy network

```python
class BLBStage2Policy(nn.Module):
    """共享 backbone + per-layer per-block 头。"""
    def __init__(self, state_dim, num_layers, action_dims_per_layer):
        super().__init__()
        # 共享 backbone：处理全局 context
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        # 每层一个 per-layer embedding，跟 backbone 输出拼起来再喂给 head
        self.layer_emb = nn.Embedding(num_layers, 64)
        # Head：跨层共享参数，输出 [num_layers, sum(action_dims_per_layer)] logits
        self.head = nn.Sequential(
            nn.Linear(256 + 64, 256), nn.ReLU(),
            nn.Linear(256, sum(action_dims_per_layer)),
        )
        self.first_input_head = nn.Linear(256, 5)   # first-input fresh 5 levels
        self.value = nn.Linear(256, 1)              # critic
```

跨层共享 head 是关键 —— 不然 12 层 × 94 dim 的全连接太重；且语义上不同层做
相似的决策。

### 7.3 训练超参（起步建议）

| 项                | 值 / 范围        |
| ----------------- | ---------------- |
| algo              | PPO              |
| n_envs            | 4–8（vec env）   |
| rollout horizon   | 1（单步 episode）|
| total_timesteps   | 50k–200k         |
| lr                | 3e-4             |
| γ (discount)      | 1.0（单步）      |
| GAE λ             | 0.95             |
| clip_range        | 0.2              |
| ent_coef          | 0.01–0.05        |
| value_coef        | 0.5              |
| batch_size        | 64               |
| n_epochs          | 4                |
| seed              | OS 熵（噪声 RNG 已经独立，可以放心固定 torch seed） |

---

## 8. 实现路线图

### M0：基础设施（必须先做）

- [ ] 新建 `blb_stage2_rl/` 目录（**不要**改动旧 `layer_importance_evaluator.py`）
- [ ] `blb_stage2_rl/action_space.py`：
  - `load_max_sfs(profile)`（§4.4）
  - `sf_from(idx, max, levels)`
  - `action_vector_to_cfgs(action_vec, max_sfs, num_layers, gelu_deg, attn_deg)` （§4.5）
  - `cfgs_to_action_vector(...)`（反向，调试用）
- [ ] `blb_stage2_rl/env.py`：实现 `BLBStage2Env(gym.Env)`，`step` 跑一回合
- [ ] `blb_stage2_rl/reward.py`：`compute_reward` 三层优先级（§6）
- [ ] 一条端到端 smoke test：用真实 `InProcessInvoker.from_profile(...)` 跑通 1 个 episode，reward 不报错

### M1：Rescale_optimizer 真接入

- [ ] 把 `Rescale_optimizer/` 拉下来；按 [`docs/README_configs.md`](README_configs.md)
  跑 `batch_run_configs.py` 生成 `static_skeletons_<profile>.json`
- [ ] 写 `optimizer_rot_to_blb_flag` 命名映射 JSON（每个 (block, profile) 一份）
- [ ] 使用 `InProcessInvoker.from_profile(...)` 真调 `replan_with_user_actions`；验证一次 evaluate 拿到合法 JSON
- [ ] 测 invalid_chain 路径：故意用 max-2*lots 让 chain 崩

### M2：单层 / 单 block 先收敛

- [ ] 先固定除 Block 1 / Layer 0 之外所有动作为 max，只让 RL 选 Block 1 / Layer 0
  的 8+1 dims。验证 PPO 能稳定下降 cost
- [ ] 再扩到 Block 1 / 全 12 层
- [ ] 再扩到 Block 1+2+3+4+5 / Layer 0
- [ ] 最后扩到全维度 1129 dim

### M3：稳定性 / 精度约束接入

- [ ] 实现 `eval_on_probe_subset(model, ...)`：返回 `(acc_mean, acc_std, loss_mean)`
- [ ] 阈值默认从 baseline 给的精度往下浮 1pp（百分点）作为 ACC_THRESHOLD
- [ ] 稳定性阈值 = baseline acc_std × 1.5

### M4：训练 / 评估 / 落盘

- [ ] 训练日志写到 `rl_results/blb_stage2/<run_name>/` （**不要**碰旧的
  `rl_results/layer_importance_runs/`）
- [ ] 每 N steps 跑一次 deterministic eval 并存 best policy
- [ ] 训练结束后导出最终 cfg `dict {layer: cfg}`，可被
  `BLBNoiseRLBridge.apply` 直接吃下

---

## 9. 文件布局建议

```
Local_program/
├── blb_stage2_rl/                 ★ 全部新代码
│   ├── __init__.py
│   ├── action_space.py            §4 + 4.4 + 4.5
│   ├── env.py                     gym.Env 包装：reset/step/close
│   ├── reward.py                  §6 三层优先级
│   ├── policy.py                  §7.2 BLBStage2Policy
│   ├── train.py                   历史设计入口；当前运行统一走 llama_7B_LayerImportance.sh
│   ├── eval.py                    deterministic eval + 导出最佳 cfg
│   ├── max_sfs/                   profile → max sf 缓存（JSON）
│   │   └── mrpc.json
│   └── rotation_name_maps/        optimizer 名 → blb flag 名
│       ├── block1_mrpc.json
│       ├── block2_mrpc.json
│       └── ...
├── docs/
│   └── BLB_stage2_rl_spec.md      ← 本文
└── rl_results/
    └── blb_stage2/                ★ 训练落盘根目录
        └── <run_name>/
            ├── ppo_step_info.txt
            ├── tensorboard/
            └── best_cfg.pkl
```

---

## 10. 与旧版 stage 2 的隔离（强约束）

1. **代码层**：所有新代码在 `blb_stage2_rl/`；不要 import
   `layer_importance_evaluator.py` 里的 `apply_*_noise_configuration` 系列。
2. **运行时层**：训练开始前先调用 `handler.restore_layer_*_noise(...)`
   清掉所有 legacy 噪声安装；BLB install 时
   `_check_blb_legacy_conflict` 会自动校验，残留 → `RuntimeError`。
3. **数据层**：训练日志写在 `rl_results/blb_stage2/` 下，不要碰
   `rl_results/layer_importance_runs/`（旧版的目录）。
4. **CLI 层**：当前外部运行统一走
   `bash llama_7B_LayerImportance.sh run rl ...`，通过
   `--stage2-rl-variant blb_v3|legacy_v2` 切换实现，不再建议用户直接调用底层入口。

---

## 11. 验证清单（写完之后逐条过一遍）

- [ ] `BLBStage2Env.reset()` 返回 state dim 正确
- [ ] `BLBStage2Env.step(action)` 不抛异常 + 返回 (obs, reward, done=True, info)
- [ ] action 全 max → reward = baseline cost reward = 0（差分基线）
- [ ] action 全 min → reward 多半很负（精度崩了），且 priority1 触发
- [ ] invalid_chain 触发时 reward = `-INVALID_PENALTY`，env 不 crash
- [ ] 跑 100 步 PPO 无 nan / inf
- [ ] BLB / legacy 互斥校验：故意先装一个 `replace_layer_input_noise`，预期
  `BLBStage2Env.step` 抛 `RuntimeError`
- [ ] `bridge.clear()` 后 `_logits` 严格等于 baseline（|Δ|max == 0）
- [ ] 噪声真随机：连续两个相同 action 的 step，logits 应当略不同（独立 RNG，
  见 `function_handler._sample_independent_gaussian`）
- [ ] 训练 50k 步，best policy 的 cost reward > baseline 且精度差距在阈值内

---

## 12. 已知风险与开放问题

1. **Rescale_optimizer 调用开销**：训练固定走 in-process replan，避免 subprocess fork；
   后续优化重点是减少每个 episode 内重复 graph/config 调用，或让 `Rescale_optimizer`
   支持一次性 batch。

2. **首层特殊情况**：Layer 0 Block 1 的 SF/noise 槽位无效，但
   `output_truncation_k` 有效。实现物化 `noise_enabled=False` 的 K-only cfg，
   由统一 bridge 安装，并在 variance → rsqrt 前调用与其它层相同的截断器。

3. **rotation flags 是输出而非输入**：注意 §3.2 流程里 RL 出动作 → 调
   optimizer → 用 optimizer 输出反写 rotation flags → 装模型。如果你颠倒顺序
   （先装 rotation flags 再调 optimizer），会拿到错的开销。务必按伪代码顺序
   做。

4. **多 N 表的 N 选取**：每个 block 的默认 N 已经写死在
   `make_block{}_default_config(N=...)` 里（Block 1 / 5 deg=1 用 8192，其余
   16384）。RL 不需要选 N（也不该选），但要记得训练时 N 必须和
   Rescale_optimizer 里的 graph 配置一致 —— 否则噪声幅度对不上。

5. **first-input fresh 只装 layer 0**：`replace_blb_first_input_noise` 默认
   `layer_indices=[0]`。RL 给的 `first_input_sf` 也只用 1 个 SF。

6. **稳定性约束**：建议用同一个 probe 跑 3 次（`probe_subset_repeat=3`）取
   acc_std；不要每个 step 重采 probe，不然 noise 是 sample noise 不是模型
   noise。

7. **legacy 互斥的副作用**：如果用户在你训练时手动跑了旧版 stage 2，模型上
   会有 legacy 噪声 → 你的 BLB install 会报错。`BLBStage2Env.reset()` 一定要
   先把 legacy 全 restore 掉再装 BLB。

---

## 13. 快速起步代码骨架（伪代码，可直接 fork）

```python
# blb_stage2_rl/env.py
import gymnasium as gym
import numpy as np
import torch
from function_handler import ReversibleLayerHandler
from blb_rl_bridge import BLBNoiseRLBridge
from rescale_optimizer_bridge import (
    RescaleOptimizerBridge, InProcessInvoker, aggregate_optimizer_signals,
    apply_rotation_flags_to_cfg,
)
from .action_space import (
    load_max_sfs, action_vector_to_cfgs, action_dims_for_config,
)
from .reward import compute_reward, BaselineCostStats


class BLBStage2Env(gym.Env):
    def __init__(self, *, model, eval_loader, profile,
                 num_layers, gelu_degree, attn_degree,
                 acc_threshold, stability_threshold,
                 rescale_invoker, rotation_name_map):
        self.model = model
        self.eval_loader = eval_loader
        self.profile = profile
        self.num_layers = num_layers
        self.gelu_deg = gelu_degree
        self.attn_deg = attn_degree
        self.acc_th = acc_threshold
        self.stab_th = stability_threshold
        self.rotation_name_map = rotation_name_map  # {(block,layer): {opt_name: blb_flag}}

        self.handler = ReversibleLayerHandler(model)
        self.handler.replace_layer_softmax(...)   # softmax + GELU 近似先装好（前置依赖）
        self.handler.replace_layer_gelu(...)
        self.bridge = BLBNoiseRLBridge(self.handler, "model.bert.encoder.layer")
        self.rescale = RescaleOptimizerBridge(invoker=rescale_invoker)

        self.max_sfs = load_max_sfs(f"blb_stage2_rl/max_sfs/{profile}.json")
        self.dims = action_dims_for_config(num_layers, gelu_degree, attn_degree)
        self.action_space = gym.spaces.MultiDiscrete(self.dims)
        self.observation_space = gym.spaces.Box(low=-1, high=1,
                                                shape=(STATE_DIM,), dtype=np.float32)

        self.baseline = self._compute_baseline()    # §6.3

    def reset(self, *, seed=None, options=None):
        # 清掉所有 legacy（防御式）
        self.handler.restore_layer_input_noise(layer_indices=range(self.num_layers))
        # ... 其它 legacy restore ...
        return self._build_state(), {}

    def step(self, action_vec):
        cfgs, first_input_sf = action_vector_to_cfgs(
            action_vec, self.max_sfs, self.num_layers,
            self.gelu_deg, self.attn_deg)

        # 调 Rescale_optimizer
        requests = self._build_optimizer_requests(cfgs)
        opt_outputs = self.rescale.evaluate_blocks(requests)
        opt_signals = aggregate_optimizer_signals(opt_outputs)

        if opt_signals.any_invalid:
            return (self._build_state(), -INVALID_PENALTY, True, False,
                    {"invalid": True, "signals": opt_signals})

        # rotation 反写
        for cname, out in opt_outputs.items():
            block_idx, layer_idx = self._parse_config_name(cname)
            eff = out.raw.get("new_compact_config", {}).get("effective_rotations", [])
            flag_names = [self.rotation_name_map[(block_idx, self.profile)][r["name"]]
                          for r in eff if r["name"] in self.rotation_name_map[(block_idx, self.profile)]]
            apply_rotation_flags_to_cfg(cfgs[block_idx][layer_idx], flag_names)

        # apply BLB → forward → metrics → clear
        self.bridge.apply(
            first_input_sf=first_input_sf, first_input_N=16384,
            block1_cfgs=cfgs[1], block2_cfgs=cfgs[2],
            block3_cfgs=cfgs[3], block4_cfgs=cfgs[4], block5_cfgs=cfgs[5],
        )
        try:
            metrics = self._eval_on_probe()
        finally:
            self.bridge.clear()

        reward = compute_reward(metrics, opt_signals, action_vec, self.baseline,
                                acc_th=self.acc_th, stab_th=self.stab_th)
        return self._build_state(), reward, True, False, {
            "metrics": metrics, "signals": opt_signals,
        }
```

---

## 14. 给 agent 的最后提醒

- 你**没看过这份仓库的代码**。请先 `Read` 一下 §2 列的 3 个 ★ 文件的关键 section
  （能 Grep 就 Grep，不要全文读）。
- 不要发明 API 名字。所有 API 在 §2 / §5.2 都列出来了，按那个调。
- 写之前先**给一份 plan**，确认无误后再动笔。这份 spec 留了相当多设计空间
  （PPO 超参、reward 量级、是否 autoregressive policy 等），你可以自由发挥但
  要在 plan 里说明你的选择。
- 写完一个 milestone 一定要写 sanity test 验证（参考仓库里其它块的写法 —
  我们之前每加一块就会写 `*_sanity.py`，跑通再删）。
- 任何对 `function_handler.py` / `blb_rl_bridge.py` /
  `rescale_optimizer_bridge.py` 的改动都要保留向后兼容；新增字段只能加在 dataclass
  尾部（带 default 值），不要重排序。
- 旧版 stage 2 RL 不能 break。改动后需要做完整回归验证，确保所有现存测试仍通过。
- 当你疑惑 "用户想要的精度阈值是多少 / S 是多少 / acc_threshold 怎么定" 时，
  **在 spec 里给的默认值起步，跑通后跟用户确认**，不要自己拍脑袋长期跑。

祝训练顺利。
