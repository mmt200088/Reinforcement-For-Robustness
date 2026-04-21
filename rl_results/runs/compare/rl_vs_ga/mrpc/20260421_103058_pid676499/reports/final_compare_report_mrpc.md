# FINAL：RL 与 GA 对比报告

- 数据集：`mrpc`
- 生成时间：`2026-04-21 11:19:12`
- RL 运行目录：`rl_results\runs\compare\rl_vs_ga\mrpc\20260421_103058_pid676499\children\rl`
- GA 运行目录：`rl_results\runs\compare\rl_vs_ga\mrpc\20260421_103058_pid676499\children\ga`

## 指标对比

| 算法 | 评估状态 | 进程状态 | 展示来源 | 配置来源 | Loss | Acc. | F1 | Stage1 Cost | Stage2 Cost | Total Cost | Time(ms) | Feasible | dLoss% | dAcc.% | dF1% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RL | ok | completed | optimized | json | 0.340674 | 0.866471 | 0.864542 | 37.0000 | 42.0000 | 79.0000 | 124.409 | Y | -1.28%（越低越好） | 0.15% | 0.20% |
| GA | ok | completed | optimized | json | 0.348639 | 0.861520 | 0.861906 | 38.0000 | 38.0000 | 76.0000 | 148.509 | Y | -5.95%（越低越好） | -0.14% | 0.25% |

## 关键配置

- RL 结果文件：`rl_results\runs\compare\rl_vs_ga\mrpc\20260421_103058_pid676499\children\rl\final_eval\final_eval_results_mrpc.json`
- GA 结果文件：`rl_results\runs\compare\rl_vs_ga\mrpc\20260421_103058_pid676499\children\ga\final_eval\final_eval_results_mrpc.json`
- RL 选中配置来源：`json`
- GA 选中配置来源：`json`
- RL 固定的 Stage-1 GELU：`[1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1]`
- RL 固定的 Stage-1 Softmax：`[2, 2, 5, 5, 5, 2, 5, 2, 5, 5, 6, 2]`
- RL 固定配置来源：`final_eval_optimized_stage1`
- RL 选中噪声配置：`{'input_noise_scaling_factors': [22, 22, 28, 22, 26, 30, 26, 30, 28, 22, 22, 28], 'wq_noise_scaling_factors': [22, 16, 14, 18, 14, 16, 22, 16, 22, 22, 22, 20], 'wk_noise_scaling_factors': [22, 20, 22, 16, 22, 18, 20, 20, 14, 16, 22, 14], 'wv_noise_scaling_factors': [16, 14, 14, 18, 14, 20, 22, 14, 14, 22, 18, 20], 'wo_noise_scaling_factors': [20, 16, 16, 20, 16, 22, 22, 22, 18, 20, 16, 22], 'wffn1_noise_scaling_factors': [22, 16, 24, 24, 24, 24, 22, 16, 24, 24, 24, 20], 'wffn2_noise_scaling_factors': [14, 16, 14, 18, 16, 22, 22, 22, 16, 20, 22, 22]}`
- RL 噪声 cost breakdown：`{'x': 7.65, 'wq': 5.6, 'wk': 5.6499999999999995, 'wv': 5.15, 'wo': 5.75, 'wffn1': 6.6, 'wffn2': 5.6}`
- GA 固定的 Stage-1 GELU：`[1, 2, 1, 2, 1, 1, 4, 1, 4, 1, 1, 1]`
- GA 固定的 Stage-1 Softmax：`[2, 2, 4, 4, 3, 4, 3, 2, 4, 4, 4, 2]`
- GA 固定配置来源：`final_eval_optimized_stage1`
- GA 选中噪声配置：`{'input_noise_scaling_factors': [26, 30, 24, 24, 24, 24, 26, 22, 26, 28, 22, 24], 'wq_noise_scaling_factors': [14, 20, 14, 18, 18, 16, 20, 16, 22, 14, 16, 16], 'wk_noise_scaling_factors': [14, 20, 16, 20, 14, 14, 14, 14, 14, 14, 16, 16], 'wv_noise_scaling_factors': [18, 14, 16, 16, 18, 16, 22, 18, 18, 18, 14, 16], 'wo_noise_scaling_factors': [16, 18, 16, 14, 16, 14, 14, 14, 18, 14, 14, 20], 'wffn1_noise_scaling_factors': [16, 18, 16, 20, 20, 22, 16, 22, 18, 16, 20, 20], 'wffn2_noise_scaling_factors': [14, 14, 18, 18, 20, 20, 20, 20, 16, 16, 16, 22]}`
- GA 噪声 cost breakdown：`{'x': 7.5, 'wq': 5.1000000000000005, 'wk': 4.650000000000001, 'wv': 5.1000000000000005, 'wo': 4.7, 'wffn1': 5.6000000000000005, 'wffn2': 5.3500000000000005}`

## Stage-2 多次评估统计

- RL 重复评估次数：`50`
- GA 重复评估次数：`50`

| 指标 | RL 均值 | RL 标准差 | RL 方差 | RL 最小值 | RL 最大值 | GA 均值 | GA 标准差 | GA 方差 | GA 最小值 | GA 最大值 | RL-GA 均值差 | 更优方 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Loss | 0.340674 | 0.010249 | 0.000105 | 0.321152 | 0.364854 | 0.348639 | 0.014501 | 0.000210 | 0.311226 | 0.387470 | -0.007965 | RL |
| Acc. | 0.866471 | 0.008103 | 0.000066 | 0.843137 | 0.879902 | 0.861520 | 0.009596 | 0.000092 | 0.843137 | 0.889706 | 0.004951 | RL |
| F1 | 0.864542 | 0.008347 | 0.000070 | 0.841736 | 0.878685 | 0.861906 | 0.009374 | 0.000088 | 0.844067 | 0.889820 | 0.002636 | RL |
