# STAGE2：RL 与 GA 对比报告

- 数据集：`mrpc`
- 生成时间：`2026-04-12 22:11:22`
- RL 运行目录：`/var/tmp/root-home/Reinforcement-For-Robustness/rl_results/runs/compare/rl_vs_ga/mrpc/20260412_134540_pid1070214/children/rl`
- GA 运行目录：`/var/tmp/root-home/Reinforcement-For-Robustness/rl_results/runs/compare/rl_vs_ga/mrpc/20260412_134540_pid1070214/children/ga`

## 指标对比

| 算法 | 评估状态 | 进程状态 | 展示来源 | 配置来源 | Loss | Acc. | F1 | Cost | Time(ms) | Feasible | dLoss% | dAcc.% | dF1% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RL | ok | completed | selected | json | 0.326834 | 0.878431 | 0.876359 | 37.7500 | 68.882 | Y | -3.75%（越低越好） | -0.17% | -0.12% |
| GA | ok | completed | selected | search | 0.333240 | 0.873529 | 0.873740 | 33.7000 | 34.911 | N | -2.66%（越低越好） | -0.72% | -0.42% |

## 关键配置

- RL 结果文件：`/var/tmp/root-home/Reinforcement-For-Robustness/rl_results/runs/compare/rl_vs_ga/mrpc/20260412_134540_pid1070214/children/rl/stage2_noise_final_eval/noise_final_eval_results_mrpc.json`
- GA 结果文件：`/var/tmp/root-home/Reinforcement-For-Robustness/rl_results/runs/compare/rl_vs_ga/mrpc/20260412_134540_pid1070214/children/ga/stage2_noise_final_eval/noise_final_eval_results_mrpc.json`
- RL 选中配置来源：`json`
- GA 选中配置来源：`search`
- RL 固定的 Stage-1 GELU：`[1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1]`
- RL 固定的 Stage-1 Softmax：`[2, 2, 5, 5, 5, 2, 5, 2, 5, 5, 6, 2]`
- RL 固定配置来源：`stage1_final_eval_selected`
- RL 选中噪声配置：`{'input_noise_scaling_factors': [28, 26, 22, 26, 22, 26, 24, 26, 28, 24, 22, 22], 'wq_noise_scaling_factors': [16, 22, 16, 14, 14, 18, 16, 14, 22, 20, 20, 14], 'wk_noise_scaling_factors': [14, 18, 14, 14, 14, 16, 20, 16, 18, 14, 14, 16], 'wv_noise_scaling_factors': [18, 14, 14, 18, 14, 20, 14, 18, 14, 16, 20, 14], 'wo_noise_scaling_factors': [14, 22, 22, 20, 14, 14, 18, 18, 20, 14, 16, 14], 'wffn1_noise_scaling_factors': [16, 22, 18, 24, 16, 16, 22, 16, 16, 22, 22, 16], 'wffn2_noise_scaling_factors': [14, 14, 14, 14, 20, 22, 14, 22, 18, 14, 14, 14]}`
- RL 噪声 cost breakdown：`{'x': 7.4, 'wq': 5.15, 'wk': 4.700000000000001, 'wv': 4.8500000000000005, 'wo': 5.15, 'wffn1': 5.6499999999999995, 'wffn2': 4.85}`
- GA 固定的 Stage-1 GELU：`[1, 2, 1, 2, 1, 1, 4, 1, 4, 1, 1, 1]`
- GA 固定的 Stage-1 Softmax：`[2, 2, 4, 4, 3, 4, 3, 2, 4, 4, 4, 2]`
- GA 固定配置来源：`stage1_final_eval_selected`
- GA 选中噪声配置：`{'input_noise_scaling_factors': [22, 22, 22, 22, 22, 24, 24, 22, 22, 24, 22, 22], 'wq_noise_scaling_factors': [14, 16, 14, 16, 14, 14, 16, 14, 18, 14, 14, 14], 'wk_noise_scaling_factors': [14, 14, 14, 16, 14, 14, 14, 16, 14, 14, 14, 14], 'wv_noise_scaling_factors': [14, 14, 14, 16, 14, 16, 14, 16, 18, 14, 16, 14], 'wo_noise_scaling_factors': [14, 14, 14, 16, 14, 14, 18, 14, 14, 16, 14, 16], 'wffn1_noise_scaling_factors': [16, 20, 16, 16, 16, 16, 16, 16, 18, 16, 16, 16], 'wffn2_noise_scaling_factors': [14, 14, 14, 14, 14, 14, 16, 14, 14, 14, 14, 16]}`
- GA 噪声 cost breakdown：`{'x': 6.75, 'wq': 4.45, 'wk': 4.300000000000001, 'wv': 4.5, 'wo': 4.450000000000001, 'wffn1': 4.950000000000001, 'wffn2': 4.300000000000001}`
