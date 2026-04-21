# STAGE1：RL 与 GA 对比报告

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
- RL GELU：`[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]`
- RL Softmax：`[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]`
- GA GELU：`[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]`
- GA Softmax：`[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]`

## 警告

- RL 的 final-eval 文件缺失，已按声明的 json 配置补做统一最终评估。
- GA 的 final-eval 文件缺失，已按声明的 json 配置补做统一最终评估。
- RL: 未获得正常的 selected 结果，当前展示已回退到 baseline。
- GA: 未获得正常的 selected 结果，当前展示已回退到 baseline。
