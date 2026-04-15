# STAGE1：RL 与 GA 对比报告

- 数据集：`mrpc`
- 生成时间：`2026-04-13 21:40:25`
- RL 运行目录：`/var/tmp/root-home/Reinforcement-For-Robustness/rl_results/runs/compare/rl_vs_ga/mrpc/20260413_213848_pid1143993/children/rl`
- GA 运行目录：`/var/tmp/root-home/Reinforcement-For-Robustness/rl_results/runs/compare/rl_vs_ga/mrpc/20260413_213848_pid1143993/children/ga`

## 指标对比

| 算法 | 评估状态 | 进程状态 | 展示来源 | 配置来源 | Loss | Acc. | F1 | Cost | Time(ms) | Feasible | dLoss% | dAcc.% | dF1% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RL | ok | completed | selected | json | 0.333366 | 0.882353 | 0.880418 | 37.0000 | 29.326 | Y | -2.97%（越低越好） | 0.28% | 0.34% |
| GA | ok | completed | selected | json | 0.344901 | 0.877451 | 0.877702 | 38.0000 | 33.439 | Y | -2.62%（越低越好） | -0.28% | 0.03% |

## 关键配置

- RL 结果文件：`/var/tmp/root-home/Reinforcement-For-Robustness/rl_results/runs/compare/rl_vs_ga/mrpc/20260413_213848_pid1143993/children/rl/stage1_final_eval/final_eval_results_mrpc.json`
- GA 结果文件：`/var/tmp/root-home/Reinforcement-For-Robustness/rl_results/runs/compare/rl_vs_ga/mrpc/20260413_213848_pid1143993/children/ga/stage1_final_eval/final_eval_results_mrpc.json`
- RL 选中配置来源：`json`
- GA 选中配置来源：`json`
- RL GELU：`[1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1]`
- RL Softmax：`[2, 2, 5, 5, 5, 2, 5, 2, 5, 5, 6, 2]`
- GA GELU：`[1, 2, 1, 2, 1, 1, 4, 1, 4, 1, 1, 1]`
- GA Softmax：`[2, 2, 4, 4, 3, 4, 3, 2, 4, 4, 4, 2]`

## 警告

- RL 的 Stage-1 最终评估文件缺失，已按声明的 json 配置补做最终评估。
- GA 的 Stage-1 最终评估文件缺失，已按声明的 json 配置补做最终评估。
