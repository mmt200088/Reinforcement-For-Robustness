# STAGE2：RL 与 GA 对比报告

- 数据集：`mrpc`
- 生成时间：`2026-04-13 21:58:47`
- RL 运行目录：`/var/tmp/root-home/Reinforcement-For-Robustness/rl_results/runs/compare/rl_vs_ga/mrpc/20260413_213848_pid1143993/children/rl`
- GA 运行目录：`/var/tmp/root-home/Reinforcement-For-Robustness/rl_results/runs/compare/rl_vs_ga/mrpc/20260413_213848_pid1143993/children/ga`

## 指标对比

| 算法 | 评估状态 | 进程状态 | 展示来源 | 配置来源 | Loss | Acc. | F1 | Cost | Time(ms) | Feasible | dLoss% | dAcc.% | dF1% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RL | ok | completed | selected | json | 0.332713 | 0.878431 | 0.876372 | 37.7500 | 30.902 | Y | -3.16%（越低越好） | -0.17% | -0.12% |
| GA | ok | completed | selected | json | 0.345097 | 0.870613 | 0.870904 | 33.7000 | 56.125 | N | -2.56%（越低越好） | -1.06% | -0.75% |

## 关键配置

- RL 结果文件：`/var/tmp/root-home/Reinforcement-For-Robustness/rl_results/runs/compare/rl_vs_ga/mrpc/20260413_213848_pid1143993/children/rl/stage2_noise_final_eval/noise_final_eval_results_mrpc.json`
- GA 结果文件：`/var/tmp/root-home/Reinforcement-For-Robustness/rl_results/runs/compare/rl_vs_ga/mrpc/20260413_213848_pid1143993/children/ga/stage2_noise_final_eval/noise_final_eval_results_mrpc.json`
- RL 选中配置来源：`json`
- GA 选中配置来源：`json`
- RL 固定的 Stage-1 GELU：`[1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1]`
- RL 固定的 Stage-1 Softmax：`[2, 2, 5, 5, 5, 2, 5, 2, 5, 5, 6, 2]`
- RL 固定配置来源：`stage2_final_eval_fixed_config`
- RL 选中噪声配置：`{'input_noise_scaling_factors': [28, 26, 22, 26, 22, 26, 24, 26, 28, 24, 22, 22], 'wq_noise_scaling_factors': [16, 22, 16, 14, 14, 18, 16, 14, 22, 20, 20, 14], 'wk_noise_scaling_factors': [14, 18, 14, 14, 14, 16, 20, 16, 18, 14, 14, 16], 'wv_noise_scaling_factors': [18, 14, 14, 18, 14, 20, 14, 18, 14, 16, 20, 14], 'wo_noise_scaling_factors': [14, 22, 22, 20, 14, 14, 18, 18, 20, 14, 16, 14], 'wffn1_noise_scaling_factors': [16, 22, 18, 24, 16, 16, 22, 16, 16, 22, 22, 16], 'wffn2_noise_scaling_factors': [14, 14, 14, 14, 20, 22, 14, 22, 18, 14, 14, 14]}`
- RL 噪声 cost breakdown：`{'x': 7.4, 'wq': 5.15, 'wk': 4.700000000000001, 'wv': 4.8500000000000005, 'wo': 5.15, 'wffn1': 5.6499999999999995, 'wffn2': 4.85}`
- GA 固定的 Stage-1 GELU：`[1, 2, 1, 2, 1, 1, 4, 1, 4, 1, 1, 1]`
- GA 固定的 Stage-1 Softmax：`[2, 2, 4, 4, 3, 4, 3, 2, 4, 4, 4, 2]`
- GA 固定配置来源：`stage2_final_eval_fixed_config`
- GA 选中噪声配置：`{'input_noise_scaling_factors': [22, 22, 22, 22, 22, 24, 24, 22, 22, 24, 22, 22], 'wq_noise_scaling_factors': [14, 16, 14, 16, 14, 14, 16, 14, 18, 14, 14, 14], 'wk_noise_scaling_factors': [14, 14, 14, 16, 14, 14, 14, 16, 14, 14, 14, 14], 'wv_noise_scaling_factors': [14, 14, 14, 16, 14, 16, 14, 16, 18, 14, 16, 14], 'wo_noise_scaling_factors': [14, 14, 14, 16, 14, 14, 18, 14, 14, 16, 14, 16], 'wffn1_noise_scaling_factors': [16, 20, 16, 16, 16, 16, 16, 16, 18, 16, 16, 16], 'wffn2_noise_scaling_factors': [14, 14, 14, 14, 14, 14, 16, 14, 14, 14, 14, 16]}`
- GA 噪声 cost breakdown：`{'x': 6.75, 'wq': 4.45, 'wk': 4.300000000000001, 'wv': 4.5, 'wo': 4.450000000000001, 'wffn1': 4.950000000000001, 'wffn2': 4.300000000000001}`

## 警告

- RL 的 Stage-2 最终评估文件缺失，已按声明的 json 配置补做最终评估。
- RL Stage-2 固定的 Stage-1 配置来源：stage1_final_eval_selected
- GA 的 Stage-2 最终评估文件缺失，已按声明的 json 配置补做最终评估。
- GA Stage-2 固定的 Stage-1 配置来源：stage1_final_eval_selected

## Stage-2 多次评估统计

- RL 重复评估次数：`100`
- GA 重复评估次数：`100`

| 指标 | RL 均值 | RL 标准差 | RL 方差 | RL 最小值 | RL 最大值 | GA 均值 | GA 标准差 | GA 方差 | GA 最小值 | GA 最大值 | RL-GA 均值差 | 更优方 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Loss | 0.332713 | 0.002976 | 0.000009 | 0.326292 | 0.340006 | 0.345097 | 0.003555 | 0.000013 | 0.335206 | 0.354731 | -0.012384 | RL |
| Acc. | 0.878431 | 0.004231 | 0.000018 | 0.867647 | 0.889706 | 0.870613 | 0.005156 | 0.000027 | 0.860294 | 0.884804 | 0.007819 | RL |
| F1 | 0.876372 | 0.004352 | 0.000019 | 0.865116 | 0.888035 | 0.870904 | 0.005035 | 0.000025 | 0.860719 | 0.884683 | 0.005468 | RL |
