# `noise_rl_module.py` 中间结果 TXT 说明

本文档说明第二阶段噪声 PPO（`NoiseRLModule` / `_NoiseOptEnv`）在训练过程中落盘的 **文本类中间结果**：应当打开哪些文件、目录结构如何，以及每一段字段的含义与阅读顺序。

---

## 1. 文件落在什么位置

评估器通过 `layer_importance_evaluator.resolve_run_output_layout` 解析运行目录。在配置了 `run_output_dir` 时，噪声阶段输出位于：

| 路径（相对一次 run） | 说明 |
|----------------------|------|
| `stage2_noise/details/noise_ppo_step_info_{a}-{b}.txt` | **主日志**：按回合区间分块写入的逐步详情（块大小见下文） |
| `stage2_noise/warning.txt` | **奖励骤降告警**：若检测到滑动窗口内平均回报异常下降，会汇总告警 |
| `stage2_noise/noise_ppo_step_info.txt` | 配置项中的「锚点路径」：`noise_rl_module` 仅用其**父目录**定位 `details/` 与 `warning.txt`；**当前实现一般不会向该文件本身写入内容** |

块大小常量：`layer_importance_evaluator.STEP_INFO_CHUNK_SIZE = 510`（即文件名中 `a-b` 常为 `1-510`、`511-1020` …）。命名示例：`noise_ppo_step_info_1-510.txt`。

**阅读建议**：分析某一轮（episode）的逐步行为时，先根据回合号 \(e\)（从 1 起算）算出 \(\lfloor (e-1)/510 \rfloor\) 对应的块文件，再在文件内搜索 `--- 回合（Episode） e`。

---

## 2. 分块详情文件整体结构

每个 `noise_ppo_step_info_*.txt` 开头有一行总标题，例如：

```text
=== 噪声PPO每步信息（Noise PPO StepInfo）回合 1-510 ===
```

随后按 **回合** 分段：

```text
--- 回合（Episode） N (回合回报（EpisodeReturn）=..., 原始最终奖励（RawFinalReward）=..., 稠密奖励合计（DenseRewardTotal）=...) ---
```

- **EpisodeReturn**：该回合 PPO buffer 里各步奖励之和（最后一步为稀疏终结奖励，中间步为稠密 shaping 奖励）。
- **RawFinalReward**：仅在整回合结束时才有意义；分块里该行仍可能写出数值（非最后一层步时为占位逻辑），以每步记录里的 `原始最终奖励` 字段为准。
- **DenseRewardTotal**：环境内累计的稠密奖励 `accumulated_dense_reward`（prefix-completion 增量形式累计）。

每个回合下按层（step）依次输出多块「键值行」，一步对应一段，段空行分隔。

---

## 3. 逐步字段说明（与代码一致）

以下字段由 `_write_noise_step_info` 写出；部分为 **条件输出**（仅当字典中存在且非 `None` 时写入）。

### 3.1 索引与状态

| 键名（文件中中文标签） | 含义 |
|------------------------|------|
| **全局步数（step_global）** | `episode_id * total_layers + step`，跨回合单调递增的训练步编号 |
| **回合编号（episode_id）** | 从 0 开始的回合下标（文件中「回合（Episode） N」的 \(N\) 为 `episode_id + 1`） |
| **层索引（layer_index）** | 当前步对应的 Transformer 层下标（0 … `total_layers-1`） |
| **状态向量（state_vector）** | 策略网络输入的扁平向量，见 **§4 状态向量布局** |

### 3.2 本步选择的噪声缩放因子（离散档位的整数 ID）

即环境 `step()` 写入的 `curr_*_noise_scaling_factor`：**来自动作索引经离散映射表得到的 scaling factor**，不是连续实数。七种对应：

- 输入 **x**：`curr_input_noise_scaling_factor`
- 注意力权重：**wq / wk / wv / wo**
- FFN：**wffn1 / wffn2**

一般语义上，**因子越大表示该处噪声缩放越「保守」**（与环境中 `allowed` 集合及 `cost_map` 一致：低成本对应更强隐私/更大噪声的场景需在代码里对照映射表）。

### 3.3 策略头输出的概率分布

| 键名 | 含义 |
|------|------|
| **x_prob_dist** … **wffn2_prob_dist** | 七个 Categorical 头在**当前步**的 `softmax` 概率向量（长度 = 各头离散动作数） |

用于检查策略是否塌缩（one-hot）、是否在探索，或与实际采取的档位对照。

### 3.4 价值与成本

| 键名 | 含义 |
|------|------|
| **评论家值（critic_value）** | 当前步 Critic 对状态价值的估计标量 |
| **累计成本（accumulated_cost）** | 从回合开始到**本步结束后**的噪声方案代价累计（每层加上本层各 head 代价） |

### 3.5 截至本步的完整离散配置（列表）

以下为从第 0 层到当前层已选 scaling factor 的 **整数列表**（长度随层数增加）：

- `input_noise_config`、`wq_noise_config`、…、`wffn2_noise_config`

用于复现「当前前缀」对应的噪声配置。

### 3.6 优化器 / PPO 超参（本步记录）

| 键名 | 含义 |
|------|------|
| **当前学习率（current_lr）** | `evaluator.update_hyperparameters` 在该回合给出的学习率 |
| **当前熵系数（current_entropy_coef）** | 同上的熵正则系数 |

### 3.7 蒙特卡洛评估统计（多为回合末尾一步才有）

来源：`info["mc_eval"]`。非终止步通常无 MC 汇总，文件中可能 **整段不出现** 下列行。

| 键名 | 含义 |
|------|------|
| **mc_samples** | `num_samples`，即 MC trial 条数 |
| **mc_loss_mean / mc_loss_std** | 多次随机评估的 loss 均值与标准差 |
| **mc_metric1_mean / mc_metric1_std** | 指标 1（如 MRPC 的准确率相关指标，依数据集而定） |
| **mc_metric2_mean / mc_metric2_std** | 指标 2；`num_metrics==1` 时仍可能存在占位统计 |

### 3.8 奖励相关

| 键名 | 含义 |
|------|------|
| **步奖励（step_reward）** | 环境返回给 PPO 的 \(r_t\)：中间步为稠密奖励，最后一步为稀疏终结奖励 |
| **稠密奖励步（dense_reward_step）** | `dense_reward`：prefix-completion 增量 \(S_t - S_{t-1}\) 再乘以 shaping 标度 |
| **原始最终奖励（raw_final_reward）** | 终结步上组装的稀疏回报（经裁剪等处理前的「最终一步」语义）；中间步常为 `None` 或不写 |
| **最终选择分数（final_selection_score）** | 与奖励分解中用于排名的 `final_selection_score` 一致；主要在终结步有意义 |
| **累计稠密奖励（accumulated_dense_reward）** | 当前回合内稠密项累计 |
| **稳定性代理（stability_proxy）** | 由 MC 标准差相对参考刻度构造的稳定性标量；**主要在终结步**写入 |
| **稳定性惩罚（stability_penalty）** | 纳入回报的稳定项惩罚；**主要在终结步** |

### 3.9 Dense probe / partial 诊断（中间步常见）

来自 `_evaluate_dense_probe_once` 与 `probe_info`，用于解释稠密奖励依据：

| 键名 | 含义 |
|------|------|
| **partial_probe_loss / partial_probe_metric1 / partial_probe_metric2** | 对「当前前缀 + completion 补齐」完整配置做 **1 次** `evaluate_noise_model` 得到的 loss / 指标 |
| **partial_margin_loss / partial_margin_metric1 / partial_margin_metric2** | 相对动态约束限的归一化 margin（`_evaluate_dense_probe_once` 内公式） |
| **partial_completion_score** | 加权 margin 效用 − 违约惩罚 + 轻量成本项后的标量 \(S(\cdot)\) |
| **partial_violation_penalty** | 违反约束时的加权和 |
| **remaining_budget_ratio** | 剩余成本预算比例 \([0,1]\) |
| **completion_policy** | 补齐后缀的策略名（如 `max_suffix`：未来层用最大 scaling factor、最低噪声） |

---

## 4. 状态向量（state_vector）布局

`state_vector` 由 `_NoiseOptEnv._get_state()` 拼接，**维度随 `total_layers` 变化**。逻辑顺序为：

1. **位置 one-hot**（长度 `total_layers`）：当前 `current_layer` 位置为 1，其余 0。  
2. **标量** `cost_deviation`：累计成本相对「按层期望成本」的偏差（有裁剪）。  
3. **prev_norms**（7 维）：上一步已选缩放因子对应的归一化标度，顺序为 **x / wq / wk / wv / wo / wffn1 / wffn2**（`wffn1` 使用独立归一化映射，`wffn2` 与 generic weight 一致）。  
4. **标量** `complexity_debt`、`progress`。  
5. **七条历史序列**（各长度 `total_layers`）：本节已决各层的归一化噪声档位（未决层为 mask 常数）。  
6. **连续特征余下 9 维**：即 `get_continuous_features()` 中的 `cont[3:]`，依次为 `safe_rate_gap`、`tail_violation`、`tail_margin`（尾部风险记忆），以及 partial probe 的 `partial_completion_score`、`partial_worst_margin`、`remaining_budget_ratio`、`partial_margin_loss`、`partial_margin_metric1`、`partial_margin_metric2`。

读日志时一般 **不需要手工拆向量**；若要与网络对齐，以 `NOISE_STAGE_CONT_DIM = 12` 及 `total_layers` 在源码中核对即可。

---

## 5. `warning.txt`（奖励骤降告警）

训练循环按窗口统计 `episode_returns`；若平均回报相对上一窗口下降超过 `REWARD_DROP_WARNING_THRESHOLD`（由 `layer_importance_evaluator` 与 `noise_rl_module` 共用引入），会将条目写入 `noise_warnings`，**噪声阶段结束**时由 `_write_warning_report` 生成 `warning.txt`。

报告中每条包含：

- **类型**：告警类型标识  
- **窗口编号**：第几个 PPO 更新窗口  
- **上次平均奖励 / 本次平均奖励 / 下降幅度**  
- **涉及回合范围** 与 **`details/` 下应对照的分块文件名**

用于在回报曲线变差时快速定位到对应 `noise_ppo_step_info_*.txt` 区段。

---

## 6. 与 TXT 并存的非文本产物（便于对照）

`noise_rl_module` 还会周期性保存 **PNG**（非 TXT），默认在 `stage2_noise/progress/`：

- `noise_risk_curves_ep*.png`、`noise_confirm_curves_ep*.png` 等  

以及回合级曲线：`stage2_noise/noise_ppo_training_curve.png`、`noise_ppo_entropy_curve.png`。  
排查问题时可将 TXT 中某回合的配置与这些曲线的阶段对齐。

---

## 7. 推荐阅读顺序（实操）

1. 从训练日志确认 **episode 编号** 或 **step_global** 范围。  
2. 打开对应 **`details/noise_ppo_step_info_{a}-{b}.txt`**，定位 `Episode` 分隔行。  
3. 先看该回合 **EpisodeReturn / DenseRewardTotal**，再看 **最后一层步** 的 `mc_*`、`final_selection_score`、`raw_final_reward`。  
4. 若中间层行为异常，回看各步 **`partial_completion_score` 与 `partial_margin_*`** 是否与 `dense_reward_step` 同向变化。  
5. 若存在 **`warning.txt`**，按其指向的回合区间复查策略是否塌缩或探索过激。

---

*文档与仓库中 `noise_rl_module.py`（`_NoiseOptEnv`、`NoiseRLModule.run`、`_write_noise_step_info`）及 `layer_importance_evaluator.py`（`STEP_INFO_CHUNK_SIZE`、`NOISE_STAGE_STEP_INFO_FILE`）行为对齐；若代码变更，请以实际写入逻辑为准。*
