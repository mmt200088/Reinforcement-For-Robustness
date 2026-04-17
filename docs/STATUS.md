# 任务进度总板 / STATUS

- 项目根目录：`/var/tmp/root-home/Reinforcement-For-Robustness`
- 生成时间：`2026-04-17 22:00:20`
- 由 `tools/status_board.py` 生成（只读扫描，不修改任何结果目录）

图例：`✓ completed`, `· not_started`, `→ skipped`, `* running`, `× failed`, `? unknown`

## 1. 单任务 RL（rl_results/persistent/rl/）


| model     | dataset | slug                      | stage1_search | stage1_eval | stage2_search | stage2_eval   | runs | 最近更新                      | PID           |
| --------- | ------- | ------------------------- | ------------- | ----------- | ------------- | ------------- | ---- | ------------------------- | ------------- |
| bert-base | mrpc    | s1t0.005_s2q0.05_s2sq0.05 | → skipped     | → skipped   | · not_started | · not_started | 1    | 2026-04-14 20:35:38       | alive(63070)  |
| bert-base | stsb    | s1t0.005_s2q0.05_s2sq0.05 | ? unknown     | ? unknown   | ? unknown     | ? unknown     | 1    | 2026-04-14 02:02:40+08:00 | dead(1173707) |


## 2. 单任务 GA（rl_results/persistent/ga/）


| model     | dataset | slug                      | stage1_search | stage1_eval   | stage2_search | stage2_eval   | runs | 最近更新                      | PID         |
| --------- | ------- | ------------------------- | ------------- | ------------- | ------------- | ------------- | ---- | ------------------------- | ----------- |
| bert-base | mrpc    | s1t0.005_s2q0.05_s2sq0.05 | · not_started | · not_started | · not_started | · not_started | 1    | 2026-04-14 20:35:08+08:00 | dead(62832) |


## 3. 通用策略 General-RL（rl_results/persistent/general-rl/）

*无记录*

## 4. RL vs GA 对比（rl_results/runs/compare/rl_vs_ga/）


| dataset | 运行名    | mode            | RL状态      | GA状态      | RL S1/S2 eval | GA S1/S2 eval | 最近更新                | 报告                                                                                                                                                                                                                                                                                                                                        |
| ------- | ------ | --------------- | --------- | --------- | ------------- | ------------- | ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| mrpc    | comp_1 | -               | completed | completed | ✓/✓           | ✓/✓           | 2026-04-12 22:11:22 | [stage1_compare_report_mrpc.md](/var/tmp/root-home/Reinforcement-For-Robustness/rl_results/runs/compare/rl_vs_ga/mrpc/comp_1/reports/stage1_compare_report_mrpc.md) / [stage2_compare_report_mrpc.md](/var/tmp/root-home/Reinforcement-For-Robustness/rl_results/runs/compare/rl_vs_ga/mrpc/comp_1/reports/stage2_compare_report_mrpc.md) |
| mrpc    | comp_2 | evaluation_only | completed | completed | ✓/✓           | ✓/✓           | 2026-04-13 21:58:47 | [stage1_compare_report_mrpc.md](/var/tmp/root-home/Reinforcement-For-Robustness/rl_results/runs/compare/rl_vs_ga/mrpc/comp_2/reports/stage1_compare_report_mrpc.md) / [stage2_compare_report_mrpc.md](/var/tmp/root-home/Reinforcement-For-Robustness/rl_results/runs/compare/rl_vs_ga/mrpc/comp_2/reports/stage2_compare_report_mrpc.md) |


## 5. 一次性实验（experiment_results/）


| 名称                     | 有 run.log | 最近变动                |
| ---------------------- | --------- | ------------------- |
| block1                 | -         | 2026-03-15 22:37:01 |
| block2                 | -         | 2026-03-15 23:02:31 |
| block3                 | -         | 2026-03-15 23:19:33 |
| final_evaluation       | -         | 2026-03-21 21:21:13 |
| layer_importance_runs  | -         | 2026-03-26 18:16:47 |
| noise_final_evaluation | -         | 2026-03-23 11:02:47 |
| noise_scaling_sweep    | ✓         | 2026-03-27 23:29:17 |
| single_layer           | -         | 2026-03-15 19:29:40 |
| stepwise               | -         | 2026-03-15 21:24:49 |


---

## 字段说明

- **stage1_search / stage1_eval**：Stage-1（GELU/Softmax）搜索与最终评估进度，取自各 `metadata.json` 的 `stage_status` 字典
- **stage2_search / stage2_eval**：Stage-2（噪声）搜索与最终评估进度
- **runs**：该 slug 累计跑过多少轮（续训次数）
- **PID alive/dead**：根据 `LATEST_PID` 文件探测该训练进程是否还活着
- **RL S1/S2 eval**：compare 实验里 RL 侧两阶段最终评估是否就绪；GA 侧类推

