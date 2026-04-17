# 任务总板 / STATUS

> 只保留任务进度和当前最优结果，省略更新时间、PID 等运维字段。

## 1. 单任务 RL（rl_results/persistent/rl/）

- `bert-base / mrpc [s1t0.005_s2q0.05_s2sq0.…]`：进度 `S1 跳过/跳过；S2 未开始/未开始；S2 已有搜索日志`；当前最优 `S2 搜索 score=0.7076，cost=39.60，ep61954`
- `bert-base / stsb [s1t0.005_s2q0.05_s2sq0.…]`：进度 `S1 未知/未知；S2 未知/未知；S1 已有搜索日志，S2 已有搜索日志`；当前最优 `S2 搜索 score=0.2255，cost=39.60，ep1845`

## 2. 单任务 GA（rl_results/persistent/ga/）

- `bert-base / mrpc [s1t0.005_s2q0.05_s2sq0.…]`：进度 `S1 未开始/未开始；S2 未开始/未开始；S1 已有搜索日志，S2 已有搜索日志`；当前最优 `S2 搜索 score=1.4159，cost=36.90，gen2441/2500`

## 3. 通用策略 General-RL（rl_results/persistent/general-rl/）

_无记录_

## 4. RL vs GA 对比（rl_results/runs/compare/rl_vs_ga/）

- `mrpc / comp_1`：进度 `RL 完成，GA 完成；终评 S1 ✓/✓，S2 ✓/✓`；当前结果 `S2 RL(主=0.8784，次=0.8764，cost=37.75，1.30x)；GA(主=0.8735，次=0.8737，cost=33.70，1.46x，不可行)`
- `mrpc / comp_2`：进度 `RL 完成，GA 完成；终评 S1 ✓/✓，S2 ✓/✓`；当前结果 `S2 RL(主=0.8784，次=0.8764，cost=37.75，1.30x)；GA(主=0.8706，次=0.8709，cost=33.70，1.46x，不可行)`

## 5. 一次性实验（experiment_results/）

- `block1`：进度 `已完成`；当前结果 `异常对最多 mnli 23/30；最大正向增益 rte +0.0108`
- `block2`：进度 `已完成`；当前结果 `显著结果 21 项；最佳 cola L3/L8 GELU=low,SM=full +0.0092`
- `block3`：进度 `已完成`；当前结果 `8 个任务；最低一致性 mrpc/mnli rho=-0.6229`
- `final_evaluation`：进度 `已完成`；当前结果 `mrpc：主=0.8824，次=0.8804，cost=37.00，1.95x`
- `layer_importance_runs`：进度 `已完成`；当前结果 `共 6 项；当前最佳 mrpc：主=0.8824，次=0.8804，cost=48.60，1.01x`
- `noise_final_evaluation`：进度 `已完成`；当前结果 `mrpc：主=0.8775，次=0.8754，cost=34.05，1.43x，不可行`
- `noise_scaling_sweep`：进度 `已完成`；当前结果 `4 个任务；最佳 sst2/wq 主=0.9278，cost=48.00，factor=20`
- `single_layer`：进度 `已完成`；当前结果 `8 个任务；最佳 rte accuracy=0.7365（较 baseline +0.0108）`
- `stepwise`：进度 `已完成`；当前结果 `8 个任务；最佳峰值 sst2 accuracy=0.9278；最佳终局 sst2 accuracy=0.9278`

---

- `S1/S2` 进度格式：`搜索/终评`
- compare 的 `终评 S1/S2` 格式：`RL/GA`
- 结果展示优先级：`S2 终评 > S2 搜索 > S1 终评 > S1 搜索`
- 自动生成：`tools/status_board.py`
