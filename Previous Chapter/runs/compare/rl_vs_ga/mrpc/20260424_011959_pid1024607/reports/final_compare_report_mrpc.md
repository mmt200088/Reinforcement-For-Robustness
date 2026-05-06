# FINAL：RL 与 GA 对比报告

- 数据集：`mrpc`
- 生成时间：`2026-04-24 01:27:24`
- RL 运行目录：`/var/tmp/root-home/Reinforcement-For-Robustness/rl_results/runs/compare/rl_vs_ga/mrpc/20260424_011959_pid1024607/children/rl`
- GA 运行目录：`/var/tmp/root-home/Reinforcement-For-Robustness/rl_results/runs/compare/rl_vs_ga/mrpc/20260424_011959_pid1024607/children/ga`

## 指标对比

| 算法 | 评估状态 | 进程状态 | 展示来源 | 配置来源 | Loss | Acc. | F1 | Stage1 Cost | Stage2 Cost | Total Cost | Time(ms) | Feasible | dLoss% | dAcc.% | dF1% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RL | ok | completed | optimized | json | 0.330595 | 0.879608 | 0.877575 | 37.0000 | 38.8500 | 75.8500 | 38.046 | Y | -3.75%（越低越好） | -0.03% | 0.02% |
| GA | ok | completed | optimized | json | 0.333670 | 0.874363 | 0.874624 | 38.0000 | 36.7000 | 74.7000 | 39.558 | N | -2.86%（越低越好） | -0.63% | -0.32% |

## 关键配置

- RL 结果文件：`/var/tmp/root-home/Reinforcement-For-Robustness/rl_results/runs/compare/rl_vs_ga/mrpc/20260424_011959_pid1024607/children/rl/final_eval/final_eval_results_mrpc.json`
- GA 结果文件：`/var/tmp/root-home/Reinforcement-For-Robustness/rl_results/runs/compare/rl_vs_ga/mrpc/20260424_011959_pid1024607/children/ga/final_eval/final_eval_results_mrpc.json`
- RL 选中配置来源：`json`
- GA 选中配置来源：`json`
- RL 固定的 Stage-1 GELU：`[1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1]`
- RL 固定的 Stage-1 Softmax：`[2, 2, 5, 5, 5, 2, 5, 2, 5, 5, 6, 2]`
- RL 固定配置来源：`final_eval_optimized_stage1`
- RL 选中噪声配置：`{'input_noise_scaling_factors': [22, 22, 22, 30, 22, 22, 24, 24, 24, 24, 30, 24], 'wq_noise_scaling_factors': [16, 14, 14, 20, 22, 22, 18, 14, 14, 22, 22, 22], 'wk_noise_scaling_factors': [14, 14, 14, 14, 22, 14, 14, 14, 14, 14, 22, 14], 'wv_noise_scaling_factors': [22, 14, 14, 20, 14, 20, 18, 14, 22, 22, 22, 20], 'wo_noise_scaling_factors': [22, 16, 14, 16, 14, 18, 16, 16, 20, 22, 16, 20], 'wffn1_noise_scaling_factors': [24, 16, 18, 16, 16, 24, 20, 18, 24, 18, 16, 20], 'wffn2_noise_scaling_factors': [14, 14, 16, 14, 16, 16, 18, 22, 20, 18, 14, 16]}`
- RL 噪声 cost breakdown：`{'x': 7.249999999999998, 'wq': 5.5, 'wk': 4.6000000000000005, 'wv': 5.55, 'wo': 5.250000000000001, 'wffn1': 5.750000000000001, 'wffn2': 4.95}`
- GA 固定的 Stage-1 GELU：`[1, 2, 1, 2, 1, 1, 4, 1, 4, 1, 1, 1]`
- GA 固定的 Stage-1 Softmax：`[2, 2, 4, 4, 3, 4, 3, 2, 4, 4, 4, 2]`
- GA 固定配置来源：`final_eval_optimized_stage1`
- GA 选中噪声配置：`{'input_noise_scaling_factors': [26, 28, 26, 24, 24, 24, 28, 22, 22, 24, 22, 24], 'wq_noise_scaling_factors': [14, 20, 14, 18, 18, 16, 18, 16, 18, 14, 16, 16], 'wk_noise_scaling_factors': [14, 18, 16, 18, 16, 14, 14, 14, 14, 14, 16, 14], 'wv_noise_scaling_factors': [16, 14, 16, 16, 18, 16, 20, 16, 18, 20, 14, 14], 'wo_noise_scaling_factors': [16, 20, 16, 14, 16, 14, 14, 14, 16, 14, 14, 18], 'wffn1_noise_scaling_factors': [16, 18, 16, 20, 18, 20, 16, 20, 18, 16, 16, 20], 'wffn2_noise_scaling_factors': [14, 14, 18, 16, 18, 16, 18, 16, 16, 16, 18, 16]}`
- GA 噪声 cost breakdown：`{'x': 7.35, 'wq': 4.950000000000001, 'wk': 4.550000000000001, 'wv': 4.949999999999999, 'wo': 4.65, 'wffn1': 5.3500000000000005, 'wffn2': 4.900000000000001}`

## 警告

- RL final-eval JSON missing/stale; regenerating unified final eval from declared json config.
- GA final-eval JSON missing/stale; regenerating unified final eval from declared json config.

## Stage-2 多次评估统计

- RL 重复评估次数：`50`
- GA 重复评估次数：`50`

| 指标 | RL 均值 | RL 标准差 | RL 方差 | RL 最小值 | RL 最大值 | GA 均值 | GA 标准差 | GA 方差 | GA 最小值 | GA 最大值 | RL-GA 均值差 | 更优方 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Loss | 0.330595 | 0.003019 | 0.000009 | 0.323219 | 0.335647 | 0.333670 | 0.002210 | 0.000005 | 0.329143 | 0.339664 | -0.003074 | RL |
| Acc. | 0.879608 | 0.003849 | 0.000015 | 0.870098 | 0.887255 | 0.874363 | 0.004379 | 0.000019 | 0.862745 | 0.882353 | 0.005245 | RL |
| F1 | 0.877575 | 0.003838 | 0.000015 | 0.868781 | 0.885400 | 0.874624 | 0.004244 | 0.000018 | 0.863559 | 0.882353 | 0.002951 | RL |
