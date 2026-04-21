# STAGE2：RL 与 GA 对比报告

- 数据集：`mrpc`
- 生成时间：`2026-04-21 10:44:28`
- RL 运行目录：`/var/tmp/root-home/Reinforcement-For-Robustness/rl_results/runs/compare/rl_vs_ga/mrpc/20260421_103058_pid676499/children/rl`
- GA 运行目录：`/var/tmp/root-home/Reinforcement-For-Robustness/rl_results/runs/compare/rl_vs_ga/mrpc/20260421_103058_pid676499/children/ga`

## 指标对比

| 算法 | 评估状态 | 进程状态 | 展示来源 | 配置来源 | Loss | Acc. | F1 | Cost | Time(ms) | Feasible | dLoss% | dAcc.% | dF1% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RL | ok | completed | baseline | json | 0.345099 | 0.865196 | 0.862800 | - | 187.447 | Y | 0.00%（越低越好） | 0.00% | 0.00% |
| GA | ok | completed | baseline | json | 0.370710 | 0.862745 | 0.859743 | - | 186.971 | Y | 0.00%（越低越好） | 0.00% | 0.00% |

## 关键配置

- RL 结果文件：`/var/tmp/root-home/Reinforcement-For-Robustness/rl_results/runs/compare/rl_vs_ga/mrpc/20260421_103058_pid676499/children/rl/final_eval/final_eval_results_mrpc.json`
- GA 结果文件：`/var/tmp/root-home/Reinforcement-For-Robustness/rl_results/runs/compare/rl_vs_ga/mrpc/20260421_103058_pid676499/children/ga/final_eval/final_eval_results_mrpc.json`
- RL 选中配置来源：`json`
- GA 选中配置来源：`json`
- RL 固定的 Stage-1 GELU：`None`
- RL 固定的 Stage-1 Softmax：`None`
- RL 固定配置来源：`stage1_rl_checkpoint_missing`
- RL 选中噪声配置：`None`
- RL 噪声 cost breakdown：`None`
- GA 固定的 Stage-1 GELU：`None`
- GA 固定的 Stage-1 Softmax：`None`
- GA 固定配置来源：`stage1_ga_artifact_missing`
- GA 选中噪声配置：`None`
- GA 噪声 cost breakdown：`None`

## 警告

- RL: 未获得正常的 selected 结果，当前展示已回退到 baseline。
- GA: 未获得正常的 selected 结果，当前展示已回退到 baseline。
- RL Stage-2 对比未能解析出固定的 Stage-1 配置；当前结果可能已回退到 baseline。
- GA Stage-2 对比未能解析出固定的 Stage-1 配置；当前结果可能已回退到 baseline。
