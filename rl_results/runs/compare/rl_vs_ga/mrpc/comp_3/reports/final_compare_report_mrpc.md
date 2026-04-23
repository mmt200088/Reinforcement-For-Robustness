# FINAL：RL 与 GA 对比报告

- 数据集：`mrpc`
- 生成时间：`2026-04-23 20:55:42`
- RL 运行目录：`/var/tmp/root-home/Reinforcement-For-Robustness/rl_results/runs/compare/rl_vs_ga/mrpc/20260423_204046_pid955249/children/rl`
- GA 运行目录：`/var/tmp/root-home/Reinforcement-For-Robustness/rl_results/runs/compare/rl_vs_ga/mrpc/20260423_204046_pid955249/children/ga`

## 指标对比

| 算法 | 评估状态 | 进程状态 | 展示来源 | 配置来源 | Loss | Acc. | F1 | Stage1 Cost | Stage2 Cost | Total Cost | Time(ms) | Feasible | dLoss% | dAcc.% | dF1% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RL | ok | completed | optimized | json | 0.336860 | 0.868137 | 0.866186 | 37.0000 | 38.8500 | 75.8500 | 38.290 | N | -3.18%（越低越好） | -0.51% | -0.49% |
| GA | ok | completed | optimized | json | 0.338949 | 0.862696 | 0.863079 | 38.0000 | 36.7000 | 74.7000 | 39.755 | Y | -4.02%（越低越好） | 0.28% | 0.75% |

## 关键配置

- RL 结果文件：`/var/tmp/root-home/Reinforcement-For-Robustness/rl_results/runs/compare/rl_vs_ga/mrpc/20260423_204046_pid955249/children/rl/final_eval/final_eval_results_mrpc.json`
- GA 结果文件：`/var/tmp/root-home/Reinforcement-For-Robustness/rl_results/runs/compare/rl_vs_ga/mrpc/20260423_204046_pid955249/children/ga/final_eval/final_eval_results_mrpc.json`
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

- RL 的 final-eval 文件缺失，已按声明的 json 配置补做统一最终评估。
- GA 的 final-eval 文件缺失，已按声明的 json 配置补做统一最终评估。

## Stage-2 多次评估统计

- RL 重复评估次数：`50`
- GA 重复评估次数：`50`

| 指标 | RL 均值 | RL 标准差 | RL 方差 | RL 最小值 | RL 最大值 | GA 均值 | GA 标准差 | GA 方差 | GA 最小值 | GA 最大值 | RL-GA 均值差 | 更优方 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Loss | 0.336860 | 0.011544 | 0.000133 | 0.313283 | 0.360469 | 0.338949 | 0.010433 | 0.000109 | 0.313599 | 0.361156 | -0.002089 | RL |
| Acc. | 0.868137 | 0.008700 | 0.000076 | 0.845588 | 0.887255 | 0.862696 | 0.009932 | 0.000099 | 0.843137 | 0.887255 | 0.005441 | RL |
| F1 | 0.866186 | 0.008817 | 0.000078 | 0.844023 | 0.885692 | 0.863079 | 0.009769 | 0.000095 | 0.844355 | 0.887255 | 0.003107 | RL |
