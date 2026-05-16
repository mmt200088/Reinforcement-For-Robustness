# ADR-004: static_skeletons 作为 BLB Stage-2 RL 唯一 baseline 源

- **Status**: Accepted
- **Date**: 2026-05-14
- **Tags**: baseline, rescale-optimizer, integration

## Context

BLB Stage-2 RL 需要一个 baseline action vec 用来：
1. **Warmstart bias**：policy head 偏向 baseline（每槽位 largest 安全 SF）
2. **Reward zero-point**：cost reward 算 `bits_drop = baseline.total_bits − action.total_bits`
3. **MaxSFsTable 校准**：把 baseline 的 SF 写入 max_sfs，让 action_index = max 解码出 baseline SF

历史上 baseline 有过 3 种来源：
- **Heuristic 估算**（`HeuristicStubInvoker`）：用经验公式估 SF，不调
  Rescale_optimizer
- **Estimated all-max**：直接拿 max_sfs 表的 max，不验证可行性
- **static_skeletons archive**：Rescale_optimizer 用 graph + canonical t
  跑出来的 baseline，存在 `Rescale_optimizer/configs/<dataset>/static_skeletons_<dataset>.json`

观察到的问题：
- Heuristic baseline 经常和真实 RO baseline 不一致，导致 RL 训出来的 best
  在 final eval（真 RO）下变 invalid
- Estimated all-max 没有可行性保证；某些 SF 组合 RO 会判 invalid
- 同一个 dataset 跨 2 个 stage1 degree 配对会有 ≠1 个 baseline，需要选

## Decision

**只允许从 `static_skeletons_<dataset>.json` 加载 baseline**。具体逻辑：

1. 启动时调 `load_static_skeletons_baseline(rescale_optimizer_root,
   dataset, num_layers, gelu_per_layer, softmax_per_layer)`：
   - 对每个 (block, layer)，按 stage1 degree 找对应的 graph entry：
     - Block 1 → `block1_<dataset>` (layer 0 跳过)
     - Block 2 → `block2_<dataset>`
     - Block 3 → `block3_exp_n<softmax_degree[layer]>`
     - Block 4 → `block4`
     - Block 5 → `block5_n<gelu_degree[layer]>`
   - 抽 `cut_point_sf[0].sf`（Fresh SF）/ `propagation_deltas[*].delta`
     （Encode SF）/ `cut_point_sf[*].sf_post`（Rescale SF）
2. `static_skeletons_baseline_to_action(...)` 把抽出的 SF 写进 calibrated
   `MaxSFsTable`，让 `make_all_max_action_vector` 解码出**与 archive 完全
   一致**的 SF
3. **如果 archive 缺 / graph key 找不到 → 训练直接 abort**（不 fallback 到
   估算）

`HeuristicStubInvoker` 已删除（2026-05-14）。`InProcessInvoker.from_profile(...)`
是 hardcoded 唯一 baseline 路径。

## Alternatives considered

| Option | Why rejected |
|--------|--------------|
| Heuristic 估算继续保留 | 与真 RO 不一致，导致 final eval 出 invalid |
| 多源 baseline + RL 自选 | 等于把 baseline 不一致问题推给 PPO，没解决 |
| 仅用 estimated all-max | 没有可行性保证 |
| 让用户每次手动给一个 baseline JSON | 增加 UX 负担；研究人员不应该手算 SF |

## Consequences

**Positive**：
- baseline 一定 RO-feasible（archive 本身是 RO 跑出来的）
- baseline / final-eval / RL 训练用同一套 RO 代码 → 训练 best 在 final-eval
  下 valid 的概率大幅上升
- 调试更简单：baseline 不一致 → 锁定为 archive 加载问题，不用排查算法

**Negative / trade-offs**：
- 严依赖 `static_skeletons_<dataset>.json` 的存在 + 正确
- 新增 dataset 需要先在 Rescale_optimizer 跑出 archive；MRPC 之外目前只
  有 wnli archive，其他 GLUE 任务暂不支持
- 训练 abort 不友好，但比"训练完才发现 best 不可行"好得多

**Things to watch**：
- 如果 archive schema 变了（新版 RO）→ 加载会失败，要更新 loader
- 新加 task / model 时，**先**生成 archive，再开 RL

## References

- Code: `blb_stage2_rl/baseline_bootstrap.py::load_static_skeletons_baseline`
- Code: `blb_stage2_rl/baseline_bootstrap.py::static_skeletons_baseline_to_action`
- Schema: `docs/blb_baseline_handover_protocol.md` §0
- Archive 位置: `Rescale_optimizer/configs/<dataset>/static_skeletons_<dataset>.json`
- 删除 HeuristicStubInvoker 的 commit：见 git log around 2026-05-14
