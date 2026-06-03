# Stage-2 RL 动作改造：从「逐槽 SF」到「逐 block fusion-count」

> 设计日期 2026-06-03 · 分支 `jk_standard_rl` · 状态：已 brainstorm + grill，待落地
>
> 关联记忆：[[stage2-skeleton-driven-ssot]]、[[stage2-degree0-relu-support]]、[[stage1-stage2-binding]]

## 0. 一句话目标

把 Stage-2 RL 每个 block 的决策，从「该 block 内全部 effective SF 槽的笛卡尔积」（≈5^577 全局，cold-start 难收敛）压成「每个 block 决策 `(fusion_option, K)`」。`fusion_option` 通过一张**离线预计算的 fusion-count 映射表**自动展开成「该 block 内全部 effective SF 槽的具体 SF 动作」，再走**完全不变**的现有管线（`action_vector_to_cfgs → bridge → replan → reward`）。

本质上仍在为每个槽选 SF，只是把可选集合限制在「每个可达 fusion-count 下、噪声最小的那 (几) 个 SF 组合」里——搜索空间从「任意 SF 组合」变成「几个特定组合里选一个」。

## 1. 背景与动机

- 现状：`BLBStage2SequentialEnv` 每步（一个 `(layer, block)`）用 GTrXL 的 `max_step_dim=24` 个 per-slot 头各选一个 SF 档；动作空间极大，需要 anchor/warmstart/safe-neighbor/guarded-radius2/三套 invalid mask 等一整套机制硬撑。
- 为进一步压缩空间，曾引入 **sub-stage**（`substage_runner.py`，逐 block-type 分阶段训练，block3 冻结）。
- 本次：用 fusion-count 映射把每 block 的决策降到「几~几十个 option × 6 档 K」，于是 **放弃 sub-stage**，回到「逐个 block 决策」（保留 sequential，每 `(layer, block)` 一步）。

## 2. 已锁定的设计决策（brainstorm + grill 结论）

| # | 决策 | 选择 |
|---|------|------|
| Q1 | 映射覆盖哪些槽 | **全部 effective SF 槽**（fresh 源 + 有效 encode W/M/S + active rescale R）。 |
| Q2 | K（block 输出截断）怎么处理 | **独立第二决策**。每步动作 = `(fusion_option, K)`。枚举映射时 K 固定在 baseline。 |
| Q3 | block3 是否参与 | **冻结在 baseline**。RL 只决策 block1/2/4/5 变体，共 **7 种 block-type**。 |
| Q4 | episode 形态 | **保留 sequential**（逐 `(layer, block)`），每步动作头从 24 个 per-slot 改成 `(fusion_option, K)`。 |
| G1 | 映射怎么构建/同步 | **服务器全量构建 + push 回 git**；本地只对小 block 自校验。 |
| G2 | 怎么开关/落地 | opt-in flag `--blb-v3-fusion-count-action`（默认 off），训练 preset 打开；旧 per-slot 路径**休眠保留**。fusion 路径 = **同一 policy 类用 `max_step_dim=2` 实例化**。 |
| G3 | 探索/安全机制 | fusion 分支**停用** invalid mask（Static/Forbidden/Empirical）+ safe-neighbor + guarded-radius2；**保留** anchor + warmstart + 衰减 prior + 熵 + KL 自适应 LR。无 option 轴 curriculum。 |
| G7 | 「噪声最小」按什么算 | **按 replan 之后实际安装的噪声点**（post-override：被融合点=0、snap 后 SF、绑定 rotation 噪声）求方差。偏序做成可插拔 `NoiseOrder`，「动作提议 SF」版作为备选实现保留。 |

## 3. 组件一：fusion-count 映射表（离线预处理）

### 3.1 构建对象与「layer 无关」性质

- 7 种 block-type：`block1 / block2 / block4 / block5_n0 / block5_n1 / block5_n2 / block5_n4`（block3 不建，见 §3.5）。
- 关键性质：每个 block 的 cost/fusion **layer 无关**——`rescale_bridge.evaluate` 把 `config_name` 的 `_L<i>` 后缀剥掉后按 graph_key（如 `block5_n4`）查 baseline。所以映射表**按 block-type 建一次、跨所有 layer 共用**。
- 每层用哪种 block5/_n? / block-type，由 Stage-1 per-layer degree 决定（`block5_n{gelu_degree[layer]}`）；step schedule 已知每步的 graph_key_suffix。

### 3.2 枚举域：effective chain 槽 vs model-only 槽

每种 block-type 的 **effective SF 槽**（fresh 源 + 有效 encode + active rescale；排除 compat-extra/inactive/bound 槽，由 `_is_action_field_effective` 定义）再分两类：

- **chain 槽**（进 replan，影响 `valid/fusion_count/total_bits`）：t_new 的 fresh/rescale（来自 `skeleton_stage_map`）+ `default_block{n}_cfg_to_delta` 实际读的 encode 字段。**这些参与枚举**。
- **model-only 槽**（不进 replan，如 block2 `wv_sf`）：固定在 **max-SF（=最小噪声、且不改 fusion/cost）**，不进枚举。

> ⚠️ **rescale(R) 槽只枚举 SF 值（action index 1..levels-1），绝不枚举 index 0（=None=「丢掉这个 rescale」）。** 依据 CKKS 心智模型（CLAUDE.md 第 2 条）：RL 永不决定一个 must-exist 操作是否发生——rescale 点是固定的，**是否融合由优化器**根据 SF schedule 决定，不是 RL 显式提 None。2026-06-03 的 block1 崩溃正是因为枚举了 index 0：RL「丢掉」rescale 被优化器接受成一个「同 bits、更低噪声」的配置，反而支配了 all-max baseline。剔掉 index 0 后 all-max baseline 自然成为最低 fusion 的全局最小方差配置。（其余非 R 槽的 index 0 是合法的最低 SF，照常枚举。）实测识别 model-only 用整轴探针自动完成。

> ⚠️ encode delta **确实影响 fusion_count**（CTPT_MUL 把 SF 累加进模数链，改变某 rescale 能否被融合）。所以不能只枚举 cut_point；chain 槽里的 encode 必须进笛卡尔积。
>
> K 不进枚举（独立第二决策），枚举时固定在 baseline K。

**规模估计（chain 槽笛卡尔积，已剔 model-only）**：block1 ≈3.6K、block2 ≈0.65M、block4 ≈1.9M、block5_n4 ≈1.15M、block5_n2/n1/n0 更小；合计 ≈4–5M 次 replan。torch-free，可按 7 类型 + 多核并行；服务器一次性构建，缓存到 skeleton 再生为止。

> 默认全量枚举（最忠实于 Q1）。若服务器构建时间过长，备选优化（暂不做，YAGNI）：对 replan 输入 `(t_new, deltas)` 签名做去重 / 早停剪枝；不引入「cut_point-only」近似（已论证 encode 影响 fusion，近似不正确）。

### 3.3 噪声偏序：post-override 实际安装方差（可插拔）

```
NoiseOrder.total_variance(installed_plan) -> float   # 越小越「噪声最小」
```

- 默认实现 `SummedInstalledVariance`：枚举的某配置经 `replan` + `apply_optimizer_output_to_cfg`（含 block2 Q/K、block4 v_mask、block5 aux fresh 绑定同步）后，得到**实际安装的噪声点**；对每个非 None 噪声点查 `NOISE_VARIANCE_TABLE_BY_N[N][SF][dist]`（fresh/encoding/rescale/rotation）求和。被融合掉的点贡献 0；含优化器 snap 后的 SF 与绑定 rotation 噪声。model-only 槽以 max-SF 计入（常数项）。K 不入方差（非正态）。
- 备选实现 `SummedProposedVariance`（按动作提议 SF 求和）保留，方便以后换公式。

### 3.4 分组、min-noise 集、option 排序

1. 枚举每个 chain 配置 → `replan` → `(valid, fusion_count)`；丢弃 invalid。
2. valid 配置按 **realized `fusion_count`** 分组。
3. 每组取 **最小 `NoiseOrder.total_variance`**；保留所有达到该最小值的**不同安装后 cfg**（按安装后 cfg 去重——安装方案完全相同则等价，只留一个；总方差相同但安装方案不同则都保留，即你说的「并列噪声最小都要采用」）。容差用极小 epsilon 容浮点误差。
4. **option 排序**：全部 kept option 按 `(fusion_count 升序, total_variance 升序, total_bits 升序, 字典序)` 排序，option_id = 排序位次。**因为 rescale 不枚举 None（见 §3.2），all-max baseline 就是最低 fusion + 全局最小方差的配置 → 自然排到 option 0**，无需特殊强制。`group_min_noise_options` 末尾**断言 option 0 == baseline**（守卫：若某非 baseline 配置达到了更低 fusion 或更低安装方差，立即报错——说明心智模型被打破，需重查噪声偏序/枚举域）。warmstart 的统一 `preferred=[0, baseline_K_idx]` 因此对每种 block-type 都成立（见 §4.3）。
   - builder 仍断言 baseline 配置（每个 effective chain 槽取 max 档、rescale 取 max SF、model-only max-SF）经 replan 后 valid，且其 SF 切片 == `make_all_max_action_vector` 的对应切片。

`option = (fusion_count, tie_index)`（你举例的 `x-y`）。policy 的动作 = 该 block-type option 列表中的 index。

### 3.5 block3 冻结

block3 不枚举：其 map 退化成**仅 1 个 option = baseline**（baseline block3 cfg）。运行期 block3 步的 fusion 头被掩码到 1 个 option、K 掩码到 baseline-K——现有「每步 num_levels 掩码」天然处理「1 选项」，**无需新写 forced-step 机制**，schedule/horizon 不变。

### 3.6 K 独立性自检

构建时抽样验证：固定若干 cut_point 配置、扫 6 档 K，确认 `fusion_count` 不随 K 变（K 在 block 边界，预期独立）。结果写进构建报告。即使某处 K 影响 fusion，运行期 reward 用**真实 replan** 输出兜底，映射只负责选 SF；映射里的 fusion_count 是 nominal 标签。

### 3.7 输出格式与缓存位置

- 缓存：`blb_stage2_rl/fusion_maps/<profile>/<graph_key>.json`，**入 git**。
- 每文件（SF/K-first，遵循全仓约定）：metadata（profile、graph_key、build commit、skeleton hash、noise_order 名、max #options）+ `options: [{option_id, fusion_count, tie_index, total_variance, total_bits, slots:{field→decoded SF}, action_indices:[...]（喂 action_vector_to_cfgs）}]`。
- 全局 metadata（跨 7 类型的 `max_num_levels = max(max #options, 6)`）写一份汇总，供 policy 实例化读取。

### 3.8 构建与同步（G1）

本地写 `scripts/blb_build_fusion_count_map.py` + 单测 → push → 服务器 `SERVER_COMMAND.md` 跑全量构建（7 类型并行多核）+ 产出 JSON + HTML/JSON 构建报告（每类型 option 数、fusion_count 分布、K 独立性、耗时）→ push 回 git → 本地 pull。本地只对 block1 / block5_n0 跑一次自校验（torch-free，几千组合）确认 builder 正确 + baseline 还原 all-max。

## 4. 组件二：运行期接入（sequential，保留逐 block）

### 4.1 动作 = `(fusion_option, K)`，复用 policy（`max_step_dim=2`）

现有 `BLBStage2SequentialPolicy` 的 actor 已按 `(max_step_dim, max_num_levels)` 参数化（`logits[B, max_step_dim, max_num_levels]` + 每步 `(slot_mask, num_levels, action_level_mask)` 掩码 + per-slot warmstart 模板 + per-slot prev-action 嵌入 + 向量化 PPO replay）。

fusion 模式 **不新写动作头**，而是把同一 policy 用：
- `max_step_dim = 2`：slot0 = fusion_option，slot1 = K。
- `max_num_levels = max(跨7类型 #options, 6)`（从构建好的 map 汇总 metadata 读）。
- 每步掩码：slot0 → 该 block-type 的 `#options`；slot1 → 6 档 K。

两个 slot 是独立 categorical：log-prob = logp_fusion + logp_K，熵 = 两者之和。critic / GTrXL backbone / PPO 机制全不变。

### 4.2 env 展开（map → 现有 full SF vec）

`BLBStage2SequentialEnv` 每步收到 `(fusion_option, K)`：
1. 查 `fusion_map[graph_key_suffix].options[fusion_option].action_indices` → 该 block 全部 effective SF 槽的现有 action-index；
2. K 的 action-index 由 slot1 给；
3. 拼成现有 full SF vec 的该 block 切片（替换原 `splice_step_action_into_full_vec` 的 per-slot 写入）；
4. 之后 `evaluate_step / commit_step`（replan + `apply_optimizer_output_to_cfg` + 各绑定同步 + per-step reward + 终局 forward + 硬优先级 reward）**完全不变**。

block2 Q/K、block4 v_mask、block5 aux fresh 绑定：map 产出的是 K 侧/主侧 SF 的 action-index，展开后经现有 `action_vector_to_cfgs` + 绑定同步处理，无需改动。

### 4.3 warmstart / anchor / 停用项（G3）

- **forced-baseline anchor**（默认 60 episode）：每步强制 `(option 0, K=baseline_K_idx)` → 展开成 baseline full vec。
- **衰减 prior**：复用现有机制，统一 `preferred = [0, baseline_K_idx]`（因 §3.4 保证 option0=baseline；`BASELINE_K_BY_BLOCK` 全 13 → `K_LEVELS.index(13)=3`）。无需 per-step 扩展。
- **保留**：衰减 prior schedule、熵、KL 自适应 LR、per-step dense cost 整形、终局硬优先级 reward。
- **停用（fusion 分支不调用，代码保留供 per-slot 路径）**：`StaticInvalidLevelMask` / `ForbiddenActionMask` / `EmpiricalInvalidLevelMask`（SF 已全 valid）、safe-neighbor 变异、`GuardedRadius2Controller`。
- 精度坍塌（选高 fusion/低 SF option）仍由 reward 硬优先级 P1 兜住。

### 4.4 flag / preset / checkpoint（G2）

- 新 flag `--blb-v3-fusion-count-action`（默认 off）穿过 `rl_tune.py → layer_importance_evaluator → BLBStage2TrainConfig`。
- mrpc 训练 preset 打开它 + `--blb-v3-reward-devices 0,1,2,3` 等沿用。
- launcher 加互斥检查：`--blb-v3-fusion-count-action` 与 `--blb-v3-substage-mode` 不可同开。
- checkpoint variant 升到 `blb_v3_sequential_gtrxl_fusioncount_v1`；旧 per-slot GTrXL ckpt 不兼容（仅 `--fresh`）。

## 5. 保持不变

baseline 选取（仍 static_skeletons all-max；新增 `baseline → (option 0, K)` 解析器供 anchor/warmstart）、`action_vector_to_cfgs`、`rescale_optimizer_bridge`、**reward（硬优先级/cost 全不变）**、single-shot env / F0 scan / candidate-store / 现有测试、SF/K-first 落盘（仍记录展开后的 full SF vec，额外附 compact `(option, K)`）、diagnostics/experiments 注册（额外记 option + nominal fusion_count + K）。

## 6. 放弃/休眠

- sub-stage（`substage_env.py` / `substage_runner.py` / `--blb-v3-substage-*`）：**暂时放弃 = 文件保留休眠**，不删；launcher 互斥检查 + 文档标注「fusion-count 取代」。

## 7. 模块/文件计划

| 文件 | 改动 |
|------|------|
| `blb_stage2_rl/fusion_count_map.py` | **新增**：`NoiseOrder` 接口 + 默认/备选实现；map 加载器；`options(gk)` / `expand(gk, option_id) -> action_indices` / `baseline_option(gk)` / `max_num_levels()`。 |
| `scripts/blb_build_fusion_count_map.py` | **新增**：离线枚举 + replan + 安装方差 + 分组 + 去重 + 排序 + 缓存 + K 独立性自检 + HTML/JSON 报告；多核并行。 |
| `blb_stage2_rl/fusion_maps/mrpc/*.json` | **新增**（服务器产物，入 git）。 |
| `blb_stage2_rl/action_space.py` | **新增** fusion 模式 step schedule（每步 2 slot：`[fusion(#options), K(6)]`）+ 展开 helper；不动现有 per-slot schedule。 |
| `blb_stage2_rl/sequential_env.py` | fusion 分支：用 map 展开 `(option, K)` → full SF vec 切片，替代 per-slot splice。 |
| `blb_stage2_rl/sequential_runner.py` | 加载 map；fusion 分支构建 schedule/掩码/`preferred=[0,k_idx]`；`max_step_dim=2` 实例化 policy；停用 mask/radius2；ckpt variant；flag 分发。 |
| `blb_stage2_rl/sequential_policy.py` | 基本不动（已参数化）；必要时按 flag 的 prev-action 嵌入/掩码细节小调。 |
| `rl_tune.py` / `layer_importance_evaluator.py` / launcher | 新 flag 穿透 + 互斥检查 + preset。 |
| `tests/test_blb_fusion_count_map.py` | **新增**（torch-free）：NoiseOrder、min-noise 去重、option0=baseline、`expand` 往返、block3=1-option、K 独立性 stub。 |

## 8. 验证阶梯（F0 → F1 → F4）

- **F0（本地 torch-free + 服务器全量）**：构建 map；自校验 baseline 还原 all-max（option0 的 action_indices == `make_all_max_action_vector` 的对应切片）；K 独立性；inspect 每类型 #options / fusion_count 分布 / 安装方差单调性。**门槛检查**：若某类型 option 数病态大（如 >几百）或只有 1 个 fusion_count，停下来复审而非硬跑。
- **F1（服务器短 sequential smoke）**：fusion-mode 200–500 episode，确认 `(option,K)` 展开正确、reward 正常、entropy/clip/KL 健康、anchor 期还原 baseline、四卡 reward probe 仍工作。
- **F4**：长跑后取 best option 序列 → 真实 BLB final eval。

全程：本地改码 → push → 服务器 `SERVER_COMMAND.md` 跑 → artifacts 回传 → 本地 pull。

## 9. 风险与待经验确认

1. **option 集规模**未知，构建后才知（F0 门槛检查兜底）。
2. **K 独立性**：构建自检；若不成立则 reward 用真实 replan 兜底，必要时把 K 折进 map（回到 Q2 备选）。
3. **构建耗时**：4–5M replan；并行后预计可控，超时再加去重/剪枝。
4. **精度坍塌**：fusion option 含低 SF；靠 anchor + warmstart + 硬优先级 reward 控制，沿用现有判健康标准。
5. **安装方差口径**：依赖 `apply_optimizer_output_to_cfg` 的 override 与 rotation 安装语义；builder 复用同一套逻辑，单测对齐。
