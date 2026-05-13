# 服务器真实调用：MRPC static_skeletons baseline 接口

## 调用信息
- 数据集：mrpc
- 远程目录：`/var/tmp/root-home/Reinforcement-For-Robustness`
- Python：`/var/tmp/root-home/miniconda3/envs/llm_ist/bin/python`
- Git HEAD：`6341ceab2bb15cd6e4cb0b98805bc88d7343a984`
- Git status（为避免和乱码混淆，未跟踪标记显示为 UNTRACKED）：
```text
## jk_standard_rl...origin/jk_standard_rl
 M Paean/blb_action_eval.py
 M "Parting Chapter/persistent/rl/bert-base/mrpc/LATEST_PID"
 M "Parting Chapter/persistent/rl/bert-base/mrpc/LATEST_RUN_DIR"
 M Rescale_optimizer/rescale_optimizer/__init__.py
 M blb_stage2_rl/action_space.py
 M blb_stage2_rl/candidate_store.py
 M blb_stage2_rl/persistence.py
 M blb_stage2_rl/policy.py
 M blb_stage2_rl/reward.py
 M blb_stage2_rl/runner.py
 D "docs/\346\200\273\346\235\277\346\233\264\346\226\260\345\221\275\344\273\244.md"
 D reports/blb_entrypoints_grep.txt
 D reports/repo_code_config_files.txt
 D reports/repo_file_list.txt
 M scripts/blb_eval_action.py
 M scripts/blb_export_action_registry.py
 M tests/test_blb_stage2_rl_regressions.py
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00491_s2t0.00491_s2st0.00491/logs/error_summary.txt"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00491_s2t0.00491_s2st0.00491/metadata.json"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00491_s2t0.00491_s2st0.00491/rl.pid"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00491_s2t0.00491_s2st0.00491/run.pid"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00491_s2t0.00491_s2st0.00491/stage1/pruning_search_log.txt"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00492_s2t0.00492_s2st0.00492/logs/error_summary.txt"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00492_s2t0.00492_s2st0.00492/metadata.json"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00492_s2t0.00492_s2st0.00492/rl.pid"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00492_s2t0.00492_s2st0.00492/run.pid"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00492_s2t0.00492_s2st0.00492/stage1/pruning_search_log.txt"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00492_s2t0.00492_s2st0.00492/stage2_noise/progress/blb_stage2_baseline_action_full.json"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00492_s2t0.00492_s2st0.00492/stage2_noise/progress/blb_stage2_baseline_action_full.md"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00492_s2t0.00492_s2st0.00492/stage2_noise/progress/blb_stage2_best_action_full.json"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00492_s2t0.00492_s2st0.00492/stage2_noise/progress/blb_stage2_best_action_full.md"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00492_s2t0.00492_s2st0.00492/stage2_noise/progress/blb_stage2_status.json"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00492_s2t0.00492_s2st0.00492/stage2_noise/pruning_search_log.txt"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00493_s2t0.00493_s2st0.00493/best_policy/constraint_metadata.json"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00493_s2t0.00493_s2st0.00493/metadata.json"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00493_s2t0.00493_s2st0.00493/rl.pid"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00493_s2t0.00493_s2st0.00493/run.pid"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00493_s2t0.00493_s2st0.00493/stage1/pruning_search_log.txt"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00493_s2t0.00493_s2st0.00493/stage2_noise/details/noise_ppo_step_info_1-360.txt"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00493_s2t0.00493_s2st0.00493/stage2_noise/progress/blb_stage2_baseline_action_full.json"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00493_s2t0.00493_s2st0.00493/stage2_noise/progress/blb_stage2_baseline_action_full.md"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00493_s2t0.00493_s2st0.00493/stage2_noise/progress/blb_stage2_best_action_full.json"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00493_s2t0.00493_s2st0.00493/stage2_noise/progress/blb_stage2_best_action_full.md"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00493_s2t0.00493_s2st0.00493/stage2_noise/progress/blb_stage2_best_cfg.pkl"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00493_s2t0.00493_s2st0.00493/stage2_noise/progress/blb_stage2_episode_trace.csv"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00493_s2t0.00493_s2st0.00493/stage2_noise/progress/blb_stage2_report.md"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00493_s2t0.00493_s2st0.00493/stage2_noise/progress/blb_stage2_rl_checkpoint_final.pt"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00493_s2t0.00493_s2st0.00493/stage2_noise/progress/blb_stage2_rl_checkpoint_live.pt"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00493_s2t0.00493_s2st0.00493/stage2_noise/progress/blb_stage2_status.json"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00493_s2t0.00493_s2st0.00493/stage2_noise/progress/blb_stage2_training_curve.npz"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00493_s2t0.00493_s2st0.00493/stage2_noise/progress/blb_stage2_training_curve.png"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00493_s2t0.00493_s2st0.00493/stage2_noise/pruning_search_log.txt"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00494_s2t0.00494_s2st0.00494/best_policy/constraint_metadata.json"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00494_s2t0.00494_s2st0.00494/metadata.json"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00494_s2t0.00494_s2st0.00494/rl.pid"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00494_s2t0.00494_s2st0.00494/run.pid"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00494_s2t0.00494_s2st0.00494/stage1/pruning_search_log.txt"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00494_s2t0.00494_s2st0.00494/stage2_noise/details/noise_ppo_step_info_1-360.txt"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00494_s2t0.00494_s2st0.00494/stage2_noise/progress/blb_stage2_baseline_action_full.json"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00494_s2t0.00494_s2st0.00494/stage2_noise/progress/blb_stage2_baseline_action_full.md"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00494_s2t0.00494_s2st0.00494/stage2_noise/progress/blb_stage2_best_action_full.json"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00494_s2t0.00494_s2st0.00494/stage2_noise/progress/blb_stage2_best_action_full.md"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00494_s2t0.00494_s2st0.00494/stage2_noise/progress/blb_stage2_best_cfg.pkl"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00494_s2t0.00494_s2st0.00494/stage2_noise/progress/blb_stage2_episode_trace.csv"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00494_s2t0.00494_s2st0.00494/stage2_noise/progress/blb_stage2_report.md"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00494_s2t0.00494_s2st0.00494/stage2_noise/progress/blb_stage2_rl_checkpoint_final.pt"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00494_s2t0.00494_s2st0.00494/stage2_noise/progress/blb_stage2_rl_checkpoint_live.pt"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00494_s2t0.00494_s2st0.00494/stage2_noise/progress/blb_stage2_status.json"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00494_s2t0.00494_s2st0.00494/stage2_noise/progress/blb_stage2_training_curve.npz"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00494_s2t0.00494_s2st0.00494/stage2_noise/progress/blb_stage2_training_curve.png"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00494_s2t0.00494_s2st0.00494/stage2_noise/pruning_search_log.txt"
UNTRACKED "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.00494_s2t0.00494_s2st0.00494/stage2_noise/warning.txt"
UNTRACKED README_FOR_GPT55PRO.md
UNTRACKED Rescale_optimizer/README_replan_interface.md
UNTRACKED Rescale_optimizer/configs/mrpc/ss_scale.json
UNTRACKED Rescale_optimizer/rescale_optimizer/replan_interface.py
UNTRACKED Rescale_optimizer/scripts/test_replan_interface.py
UNTRACKED _sync_metadata/diff_hash.txt
UNTRACKED _sync_metadata/git_diff.patch
UNTRACKED _sync_metadata/git_diff_stat.txt
UNTRACKED _sync_metadata/git_head.txt
UNTRACKED _sync_metadata/git_status.txt
UNTRACKED _sync_metadata/package_manifest.txt
UNTRACKED _sync_metadata/result_code_alignment.txt
UNTRACKED _sync_metadata/sha256sums.txt
UNTRACKED blb_stage2_rl/action_mask.py
UNTRACKED blb_stage2_rl/baseline_bootstrap.py
UNTRACKED blb_stage2_rl/feasibility.py
UNTRACKED reports/blb_opt/trust0_examples/final_eval_feasibility.json
UNTRACKED reports/blb_opt/trust0_examples/final_eval_feasibility.md
UNTRACKED reports/blb_opt/trust0_f0_baseline/candidates/candidate_store.jsonl
UNTRACKED reports/blb_opt/trust0_f0_baseline/optimizer_outputs.json
UNTRACKED reports/blb_opt/trust0_f0_baseline/rank_key.json
UNTRACKED reports/blb_opt/trust0_f0_baseline_stage1ctx/candidates/candidate_store.jsonl
UNTRACKED reports/blb_opt/trust0_f0_baseline_stage1ctx/optimizer_outputs.json
UNTRACKED reports/blb_opt/trust0_f0_baseline_stage1ctx/rank_key.json
UNTRACKED reports/blb_opt/trust0_manifest/run_manifest.json
UNTRACKED reports/blb_opt/trust0_manifest/run_manifest.md
UNTRACKED reports/blb_opt/trust0_manifest/server_run_manifest.json
UNTRACKED reports/blb_opt/trust0_manifest/server_run_manifest.md
UNTRACKED reports/blb_opt/trust0_registry/action_index_mapping.md
UNTRACKED reports/blb_opt/trust0_registry/current_code_action_registry.json
UNTRACKED reports/blb_opt/trust0_registry/current_code_action_registry.md
UNTRACKED reports/blb_opt/trust0_registry/current_code_slot_check.md
UNTRACKED reports/blb_opt/trust0_registry/current_code_slot_semantics.md
UNTRACKED reports/blb_opt/trust0_registry/rotation_derived_slots.md
UNTRACKED reports/blb_opt/trust0_registry/slot_registry_effective.json
UNTRACKED reports/server_code_sync_report.json
UNTRACKED reports/server_code_sync_report.md
UNTRACKED reports/server_environment_snapshot.txt
UNTRACKED reports/trust0_handoff_zip_info.txt
UNTRACKED reports/trust0_remote_test_log.txt
UNTRACKED reports/trust0_smoke_artifacts/s1t0.00494_s2t0.00494_s2st0.00494/stage2_noise__details__noise_ppo_step_info_1-360.txt
UNTRACKED reports/trust0_smoke_artifacts/s1t0.00494_s2t0.00494_s2st0.00494/stage2_noise__progress__blb_stage2_baseline_action_full.json
UNTRACKED reports/trust0_smoke_artifacts/s1t0.00494_s2t0.00494_s2st0.00494/stage2_noise__progress__blb_stage2_best_action_full.json
UNTRACKED reports/trust0_smoke_artifacts/s1t0.00494_s2t0.00494_s2st0.00494/stage2_noise__progress__blb_stage2_report.md
UNTRACKED reports/trust0_smoke_artifacts/s1t0.00494_s2t0.00494_s2st0.00494/stage2_noise__progress__blb_stage2_status.json
UNTRACKED reports/trust0_smoke_artifacts/s1t0.00494_s2t0.00494_s2st0.00494/stage2_noise__warning.txt
UNTRACKED reports/trust0_smoke_summary.json
UNTRACKED reports/trust0_smoke_summary.md
UNTRACKED scripts/blb_make_run_manifest.py
UNTRACKED tests/test_blb_action_mask.py
UNTRACKED tests/test_blb_candidate_store_identity.py
UNTRACKED tests/test_blb_cost_semantics.py
UNTRACKED tests/test_blb_eval_action_stage1_context.py
UNTRACKED tests/test_blb_final_eval_feasibility.py
UNTRACKED tests/test_blb_threshold_semantics.py
UNTRACKED tests/test_blb_warmstart_resume.py
```
- Stage1 GELU degree：[1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1]
- Stage1 softmax degree：[2, 2, 5, 5, 5, 2, 5, 2, 5, 5, 6, 2]
- archive：`/var/tmp/root-home/Reinforcement-For-Robustness/Rescale_optimizer/configs/mrpc/static_skeletons_mrpc.json`
- 调用：`load_static_skeletons_baseline(...)` + `static_skeletons_baseline_to_action(..., snap_sf_to_noise_table=False)` + `describe_action_vector(...)`
- action 向量长度：877
- 读取到的 `(block, layer)`：59 / 59
- baseline total_bits_sum：14779
- avg_k：13.0
- archive SF 与实际 decode 不一致数：0

## 统计
- 无效槽：layer0 block1 不安装: 9
- 来自 static_skeletons：fresh（固定 fresh 噪声）: 59
- archive 未显式给出：保留动作空间默认 max: 72
- 来自 static_skeletons：encode（操作数噪声）: 225
- static_skeletons 未出现 sf_post/drop：rescale=OFF: 294
- 来自 static_skeletons：rescale（sf_post/drop）: 158
- RO baseline 不返回 K：默认最大 k: 59
- 无效槽：first_input 已废弃: 1

## 逐层 graph_key
| layer | GELU | softmax | block1 | block2 | block3 | block4 | block5 |
|---:|---:|---:|---|---|---|---|---|
| 0 | 1 | 2 | 跳过（layer0 不安装） | block2_mrpc | block3_exp_n2 | block4 | block5_n1 |
| 1 | 1 | 2 | block1_mrpc | block2_mrpc | block3_exp_n2 | block4 | block5_n1 |
| 2 | 1 | 5 | block1_mrpc | block2_mrpc | block3_exp_n5 | block4 | block5_n1 |
| 3 | 1 | 5 | block1_mrpc | block2_mrpc | block3_exp_n5 | block4 | block5_n1 |
| 4 | 1 | 5 | block1_mrpc | block2_mrpc | block3_exp_n5 | block4 | block5_n1 |
| 5 | 4 | 2 | block1_mrpc | block2_mrpc | block3_exp_n2 | block4 | block5_n4 |
| 6 | 1 | 5 | block1_mrpc | block2_mrpc | block3_exp_n5 | block4 | block5_n1 |
| 7 | 1 | 2 | block1_mrpc | block2_mrpc | block3_exp_n2 | block4 | block5_n1 |
| 8 | 1 | 5 | block1_mrpc | block2_mrpc | block3_exp_n5 | block4 | block5_n1 |
| 9 | 1 | 5 | block1_mrpc | block2_mrpc | block3_exp_n5 | block4 | block5_n1 |
| 10 | 1 | 6 | block1_mrpc | block2_mrpc | block3_exp_n6 | block4 | block5_n1 |
| 11 | 1 | 2 | block1_mrpc | block2_mrpc | block3_exp_n2 | block4 | block5_n1 |

## 完整动作表（真实 SF / OFF / k）
| idx | layer | block | graph_key | 槽位 | 值 | 生效 | archive_sf | 来源 | 备注 |
|---:|---:|---|---|---|---|---|---:|---|---|
| 0 | 0 | 1 | block1_mrpc | Block1：GELU 输出 fresh 噪声 (`gelu_out_sf`) | SF=30 | False |  | 无效槽：layer0 block1 不安装 | layer 0 has no upstream FFN2; the first HE config is treated as lossless so block1 noise is not installed (aligned with Rescale_optimizer) |
| 1 | 0 | 1 | block1_mrpc | Block1：Wffn2 操作数 encode 噪声 (`wffn2_sf`) | SF=20 | False |  | 无效槽：layer0 block1 不安装 | layer 0 has no upstream FFN2; the first HE config is treated as lossless so block1 noise is not installed (aligned with Rescale_optimizer) |
| 2 | 0 | 1 | block1_mrpc | Block1：第一个 1/d 操作数 encode 噪声 (`mean_inv_d_sf`) | SF=20 | False |  | 无效槽：layer0 block1 不安装 | layer 0 has no upstream FFN2; the first HE config is treated as lossless so block1 noise is not installed (aligned with Rescale_optimizer) |
| 3 | 0 | 1 | block1_mrpc | Block1：第二个 1/d 操作数 encode 噪声 (`var_inv_d_sf`) | SF=20 | False |  | 无效槽：layer0 block1 不安装 | layer 0 has no upstream FFN2; the first HE config is treated as lossless so block1 noise is not installed (aligned with Rescale_optimizer) |
| 4 | 0 | 1 | block1_mrpc | Block1：Wffn2*X 后结果 rescale 噪声 (`wffn2_rescale_sf`) | OFF（不加 rescale 噪声） | False |  | 无效槽：layer0 block1 不安装 | layer 0 has no upstream FFN2; the first HE config is treated as lossless so block1 noise is not installed (aligned with Rescale_optimizer) |
| 5 | 0 | 1 | block1_mrpc | Block1：第一个 1/d 乘法后结果 rescale 噪声 (`mean_rescale_sf`) | OFF（不加 rescale 噪声） | False |  | 无效槽：layer0 block1 不安装 | layer 0 has no upstream FFN2; the first HE config is treated as lossless so block1 noise is not installed (aligned with Rescale_optimizer) |
| 6 | 0 | 1 | block1_mrpc | Block1：(X-u)^2 后结果 rescale 噪声 (`square_rescale_sf`) | OFF（不加 rescale 噪声） | False |  | 无效槽：layer0 block1 不安装 | layer 0 has no upstream FFN2; the first HE config is treated as lossless so block1 noise is not installed (aligned with Rescale_optimizer) |
| 7 | 0 | 1 | block1_mrpc | Block1：第二个 1/d 乘法后结果 rescale 噪声 (`var_rescale_sf`) | OFF（不加 rescale 噪声） | False |  | 无效槽：layer0 block1 不安装 | layer 0 has no upstream FFN2; the first HE config is treated as lossless so block1 noise is not installed (aligned with Rescale_optimizer) |
| 8 | 0 | 1 | block1_mrpc | Block1：输出截断 k (`output_truncation_k`) | k=13 | False |  | 无效槽：layer0 block1 不安装 | layer 0 has no upstream FFN2; the first HE config is treated as lossless so block1 noise is not installed (aligned with Rescale_optimizer) |
| 9 | 0 | 2 | block2_mrpc | Block2：inv_std/std 相关输入 fresh 噪声 (`inv_std_fresh_sf`) | SF=31 | True | 31 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 10 | 0 | 2 | block2_mrpc | Block2：X_centered 输入 fresh 噪声 (`x_centered_fresh_sf`) | SF=30 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 11 | 0 | 2 | block2_mrpc | Block2：LayerNorm gamma 操作数 encode 噪声 (`gamma_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 12 | 0 | 2 | block2_mrpc | Block2：Wq 操作数 encode 噪声（RO ctpt_wq_wk 共享） (`wq_sf`) | SF=22 | True | 22 | 来自 static_skeletons：encode（操作数噪声） |  |
| 13 | 0 | 2 | block2_mrpc | Block2：Wk 操作数 encode 噪声（RO ctpt_wq_wk 共享） (`wk_sf`) | SF=22 | True | 22 | 来自 static_skeletons：encode（操作数噪声） |  |
| 14 | 0 | 2 | block2_mrpc | Block2：Wv 操作数 encode 噪声 (`wv_sf`) | SF=22 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 15 | 0 | 2 | block2_mrpc | Block2：KT 第一个 mask 操作数 encode 噪声 (`kt_mask1_sf`) | SF=15 | True | 15 | 来自 static_skeletons：encode（操作数噪声） |  |
| 16 | 0 | 2 | block2_mrpc | Block2：KT 第二个 mask 操作数 encode 噪声 (`kt_mask2_sf`) | SF=15 | True | 15 | 来自 static_skeletons：encode（操作数噪声） |  |
| 17 | 0 | 2 | block2_mrpc | Block2：Q 第一个 mask 操作数 encode 噪声 (`q_mask1_sf`) | SF=15 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 18 | 0 | 2 | block2_mrpc | Block2：Q 第二个 mask 操作数 encode 噪声 (`q_mask2_sf`) | SF=15 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 19 | 0 | 2 | block2_mrpc | Block2：QKT merge mask 操作数 encode 噪声 (`qkt_merge_mask_sf`) | SF=15 | True | 15 | 来自 static_skeletons：encode（操作数噪声） |  |
| 20 | 0 | 2 | block2_mrpc | Block2：(X-u)/std 归一化后 rescale 噪声 (`normalize_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 21 | 0 | 2 | block2_mrpc | Block2：乘 gamma 后 rescale 噪声 (`gamma_rescale_sf`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 22 | 0 | 2 | block2_mrpc | Block2：WkX 后 rescale 噪声 (`wk_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 23 | 0 | 2 | block2_mrpc | Block2：WqX 后 rescale 噪声 (`wq_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 24 | 0 | 2 | block2_mrpc | Block2：WvX 后 rescale 噪声 (`wv_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 25 | 0 | 2 | block2_mrpc | Block2：KT 第一个 mask 后 rescale 噪声 (`kt_mask1_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 26 | 0 | 2 | block2_mrpc | Block2：KT 第二个 mask 后 rescale 噪声 (`kt_mask2_rescale_sf`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 27 | 0 | 2 | block2_mrpc | Block2：Q 第一个 mask 后 rescale 噪声 (`q_mask1_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 28 | 0 | 2 | block2_mrpc | Block2：Q 第二个 mask 后 rescale 噪声 (`q_mask2_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 29 | 0 | 2 | block2_mrpc | Block2：Q*KT 后 rescale 噪声 (`qkt_matmul_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 30 | 0 | 2 | block2_mrpc | Block2：QKT merge mask 后 rescale 噪声 (`qkt_merge_mask_rescale_sf`) | SF=28 | True | 28 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 31 | 0 | 2 | block2_mrpc | Block2：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 32 | 0 | 3 | block3_exp_n2 | Block3：softmax 输入 X fresh 噪声 (`x_fresh_sf`) | SF=27 | True | 27 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 33 | 0 | 3 | block3_exp_n2 | Block3：1/(2n) 操作数 encode 噪声 (`inv_2n_sf`) | SF=16 | True | 16 | 来自 static_skeletons：encode（操作数噪声） |  |
| 34 | 0 | 3 | block3_exp_n2 | Block3：X*(1/2n) 后 rescale 噪声 (`x_inv_2n_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 35 | 0 | 3 | block3_exp_n2 | Block3：softmax 平方链第 1 个 rescale 噪声 (`square_rescale_sf_0`) | SF=34 | True | 34 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 36 | 0 | 3 | block3_exp_n2 | Block3：softmax 平方链第 2 个 rescale 噪声 (`square_rescale_sf_1`) | SF=34 | True | 34 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 37 | 0 | 3 | block3_exp_n2 | Block3：softmax 平方链第 3 个 rescale 噪声 (`square_rescale_sf_2`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | softmax degree 2 does not use this square-rescale slot |
| 38 | 0 | 3 | block3_exp_n2 | Block3：softmax 平方链第 4 个 rescale 噪声 (`square_rescale_sf_3`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | softmax degree 2 does not use this square-rescale slot |
| 39 | 0 | 3 | block3_exp_n2 | Block3：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 40 | 0 | 4 | block4 | Block4：softmax 输出 fresh 噪声 (`softmax_out_fresh_sf`) | SF=35 | True | 35 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 41 | 0 | 4 | block4 | Block4：V 输入 fresh 噪声 (`v_fresh_sf`) | SF=30 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 42 | 0 | 4 | block4 | Block4：softmax mask 操作数 encode 噪声 (`softmax_out_mask_sf`) | SF=14 | True | 14 | 来自 static_skeletons：encode（操作数噪声） |  |
| 43 | 0 | 4 | block4 | Block4：V mask 操作数 encode 噪声 (`v_mask_sf`) | SF=14 | True | 14 | 来自 static_skeletons：encode（操作数噪声） |  |
| 44 | 0 | 4 | block4 | Block4：softmax*V mask 操作数 encode 噪声 (`softmax_v_mask_sf`) | SF=14 | True | 14 | 来自 static_skeletons：encode（操作数噪声） |  |
| 45 | 0 | 4 | block4 | Block4：attention LN 第一个 1/d encode 噪声 (`ln_mean_inv_d_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 46 | 0 | 4 | block4 | Block4：attention LN 第二个 1/d encode 噪声 (`ln_var_inv_d_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 47 | 0 | 4 | block4 | Block4：Wo 操作数 encode 噪声 (`wo_sf`) | SF=22 | True | 22 | 来自 static_skeletons：encode（操作数噪声） |  |
| 48 | 0 | 4 | block4 | Block4：softmax mask 后 rescale 噪声 (`softmax_out_mask_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 49 | 0 | 4 | block4 | Block4：V mask 后 rescale 噪声 (`v_mask_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 50 | 0 | 4 | block4 | Block4：softmax*V 乘法后 rescale 噪声 (`softmax_v_matmul_rescale_sf`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 51 | 0 | 4 | block4 | Block4：softmax*V mask 后 rescale 噪声 (`softmax_v_mask_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 52 | 0 | 4 | block4 | Block4：At*Wo 后 rescale 噪声 (`wo_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 53 | 0 | 4 | block4 | Block4：LN 均值分支 rescale 噪声 (`ln_mean_rescale_sf`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 54 | 0 | 4 | block4 | Block4：(X-u)^2 后 rescale 噪声 (`ln_square_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 55 | 0 | 4 | block4 | Block4：LN 方差分支 rescale 噪声 (`ln_var_rescale_sf`) | SF=28 | True | 28 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 56 | 0 | 4 | block4 | Block4：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 57 | 0 | 5 | block5_n1 | Block5：FFN LN inv_std fresh 噪声 (`inv_std_fresh_sf`) | SF=30 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 58 | 0 | 5 | block5_n1 | Block5：FFN LN x_mean/X_centered fresh 噪声 (`x_centered_fresh_sf`) | SF=30 | True | 30 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 59 | 0 | 5 | block5_n1 | Block5：FFN LN gamma 操作数 encode 噪声 (`gamma_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 60 | 0 | 5 | block5_n1 | Block5：Wffn1 操作数 encode 噪声 (`wffn1_sf`) | SF=22 | True | 22 | 来自 static_skeletons：encode（操作数噪声） |  |
| 61 | 0 | 5 | block5_n1 | Block5：GELU 系数操作数 encode 噪声 (`gelu_coeff_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 62 | 0 | 5 | block5_n1 | Block5：(X-u)/std 后 rescale 噪声 (`normalize_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 63 | 0 | 5 | block5_n1 | Block5：乘 gamma 后 rescale 噪声 (`gamma_rescale_sf`) | SF=30 | True | 30 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 64 | 0 | 5 | block5_n1 | Block5：Wffn1*X 后 rescale 噪声 (`wffn1_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 65 | 0 | 5 | block5_n1 | Block5：GELU x^2 后 rescale 噪声 (`gelu_power_rescale_sf_0`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this power-rescale slot |
| 66 | 0 | 5 | block5_n1 | Block5：GELU x^3 后 rescale 噪声 (`gelu_power_rescale_sf_1`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this power-rescale slot |
| 67 | 0 | 5 | block5_n1 | Block5：GELU x^4 后 rescale 噪声 (`gelu_power_rescale_sf_2`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this power-rescale slot |
| 68 | 0 | 5 | block5_n1 | Block5：GELU degree=1 系数乘法后 rescale 噪声 (`gelu_coeff_mul_rescale_sf_0`) | SF=30 | True | 30 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 69 | 0 | 5 | block5_n1 | Block5：GELU degree=2 系数乘法后 rescale 噪声 (`gelu_coeff_mul_rescale_sf_1`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this coefficient-rescale slot |
| 70 | 0 | 5 | block5_n1 | Block5：GELU degree=3 占位系数 rescale 噪声 (`gelu_coeff_mul_rescale_sf_2`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this coefficient-rescale slot |
| 71 | 0 | 5 | block5_n1 | Block5：GELU degree=4 系数乘法后 rescale 噪声 (`gelu_coeff_mul_rescale_sf_3`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this coefficient-rescale slot |
| 72 | 0 | 5 | block5_n1 | Block5：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 73 | 1 | 1 | block1_mrpc | Block1：GELU 输出 fresh 噪声 (`gelu_out_sf`) | SF=30 | True | 30 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 74 | 1 | 1 | block1_mrpc | Block1：Wffn2 操作数 encode 噪声 (`wffn2_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 75 | 1 | 1 | block1_mrpc | Block1：第一个 1/d 操作数 encode 噪声 (`mean_inv_d_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 76 | 1 | 1 | block1_mrpc | Block1：第二个 1/d 操作数 encode 噪声 (`var_inv_d_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 77 | 1 | 1 | block1_mrpc | Block1：Wffn2*X 后结果 rescale 噪声 (`wffn2_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 78 | 1 | 1 | block1_mrpc | Block1：第一个 1/d 乘法后结果 rescale 噪声 (`mean_rescale_sf`) | SF=34 | True | 34 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 79 | 1 | 1 | block1_mrpc | Block1：(X-u)^2 后结果 rescale 噪声 (`square_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 80 | 1 | 1 | block1_mrpc | Block1：第二个 1/d 乘法后结果 rescale 噪声 (`var_rescale_sf`) | SF=34 | True | 34 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 81 | 1 | 1 | block1_mrpc | Block1：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 82 | 1 | 2 | block2_mrpc | Block2：inv_std/std 相关输入 fresh 噪声 (`inv_std_fresh_sf`) | SF=31 | True | 31 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 83 | 1 | 2 | block2_mrpc | Block2：X_centered 输入 fresh 噪声 (`x_centered_fresh_sf`) | SF=30 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 84 | 1 | 2 | block2_mrpc | Block2：LayerNorm gamma 操作数 encode 噪声 (`gamma_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 85 | 1 | 2 | block2_mrpc | Block2：Wq 操作数 encode 噪声（RO ctpt_wq_wk 共享） (`wq_sf`) | SF=22 | True | 22 | 来自 static_skeletons：encode（操作数噪声） |  |
| 86 | 1 | 2 | block2_mrpc | Block2：Wk 操作数 encode 噪声（RO ctpt_wq_wk 共享） (`wk_sf`) | SF=22 | True | 22 | 来自 static_skeletons：encode（操作数噪声） |  |
| 87 | 1 | 2 | block2_mrpc | Block2：Wv 操作数 encode 噪声 (`wv_sf`) | SF=22 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 88 | 1 | 2 | block2_mrpc | Block2：KT 第一个 mask 操作数 encode 噪声 (`kt_mask1_sf`) | SF=15 | True | 15 | 来自 static_skeletons：encode（操作数噪声） |  |
| 89 | 1 | 2 | block2_mrpc | Block2：KT 第二个 mask 操作数 encode 噪声 (`kt_mask2_sf`) | SF=15 | True | 15 | 来自 static_skeletons：encode（操作数噪声） |  |
| 90 | 1 | 2 | block2_mrpc | Block2：Q 第一个 mask 操作数 encode 噪声 (`q_mask1_sf`) | SF=15 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 91 | 1 | 2 | block2_mrpc | Block2：Q 第二个 mask 操作数 encode 噪声 (`q_mask2_sf`) | SF=15 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 92 | 1 | 2 | block2_mrpc | Block2：QKT merge mask 操作数 encode 噪声 (`qkt_merge_mask_sf`) | SF=15 | True | 15 | 来自 static_skeletons：encode（操作数噪声） |  |
| 93 | 1 | 2 | block2_mrpc | Block2：(X-u)/std 归一化后 rescale 噪声 (`normalize_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 94 | 1 | 2 | block2_mrpc | Block2：乘 gamma 后 rescale 噪声 (`gamma_rescale_sf`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 95 | 1 | 2 | block2_mrpc | Block2：WkX 后 rescale 噪声 (`wk_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 96 | 1 | 2 | block2_mrpc | Block2：WqX 后 rescale 噪声 (`wq_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 97 | 1 | 2 | block2_mrpc | Block2：WvX 后 rescale 噪声 (`wv_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 98 | 1 | 2 | block2_mrpc | Block2：KT 第一个 mask 后 rescale 噪声 (`kt_mask1_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 99 | 1 | 2 | block2_mrpc | Block2：KT 第二个 mask 后 rescale 噪声 (`kt_mask2_rescale_sf`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 100 | 1 | 2 | block2_mrpc | Block2：Q 第一个 mask 后 rescale 噪声 (`q_mask1_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 101 | 1 | 2 | block2_mrpc | Block2：Q 第二个 mask 后 rescale 噪声 (`q_mask2_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 102 | 1 | 2 | block2_mrpc | Block2：Q*KT 后 rescale 噪声 (`qkt_matmul_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 103 | 1 | 2 | block2_mrpc | Block2：QKT merge mask 后 rescale 噪声 (`qkt_merge_mask_rescale_sf`) | SF=28 | True | 28 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 104 | 1 | 2 | block2_mrpc | Block2：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 105 | 1 | 3 | block3_exp_n2 | Block3：softmax 输入 X fresh 噪声 (`x_fresh_sf`) | SF=27 | True | 27 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 106 | 1 | 3 | block3_exp_n2 | Block3：1/(2n) 操作数 encode 噪声 (`inv_2n_sf`) | SF=16 | True | 16 | 来自 static_skeletons：encode（操作数噪声） |  |
| 107 | 1 | 3 | block3_exp_n2 | Block3：X*(1/2n) 后 rescale 噪声 (`x_inv_2n_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 108 | 1 | 3 | block3_exp_n2 | Block3：softmax 平方链第 1 个 rescale 噪声 (`square_rescale_sf_0`) | SF=34 | True | 34 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 109 | 1 | 3 | block3_exp_n2 | Block3：softmax 平方链第 2 个 rescale 噪声 (`square_rescale_sf_1`) | SF=34 | True | 34 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 110 | 1 | 3 | block3_exp_n2 | Block3：softmax 平方链第 3 个 rescale 噪声 (`square_rescale_sf_2`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | softmax degree 2 does not use this square-rescale slot |
| 111 | 1 | 3 | block3_exp_n2 | Block3：softmax 平方链第 4 个 rescale 噪声 (`square_rescale_sf_3`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | softmax degree 2 does not use this square-rescale slot |
| 112 | 1 | 3 | block3_exp_n2 | Block3：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 113 | 1 | 4 | block4 | Block4：softmax 输出 fresh 噪声 (`softmax_out_fresh_sf`) | SF=35 | True | 35 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 114 | 1 | 4 | block4 | Block4：V 输入 fresh 噪声 (`v_fresh_sf`) | SF=30 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 115 | 1 | 4 | block4 | Block4：softmax mask 操作数 encode 噪声 (`softmax_out_mask_sf`) | SF=14 | True | 14 | 来自 static_skeletons：encode（操作数噪声） |  |
| 116 | 1 | 4 | block4 | Block4：V mask 操作数 encode 噪声 (`v_mask_sf`) | SF=14 | True | 14 | 来自 static_skeletons：encode（操作数噪声） |  |
| 117 | 1 | 4 | block4 | Block4：softmax*V mask 操作数 encode 噪声 (`softmax_v_mask_sf`) | SF=14 | True | 14 | 来自 static_skeletons：encode（操作数噪声） |  |
| 118 | 1 | 4 | block4 | Block4：attention LN 第一个 1/d encode 噪声 (`ln_mean_inv_d_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 119 | 1 | 4 | block4 | Block4：attention LN 第二个 1/d encode 噪声 (`ln_var_inv_d_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 120 | 1 | 4 | block4 | Block4：Wo 操作数 encode 噪声 (`wo_sf`) | SF=22 | True | 22 | 来自 static_skeletons：encode（操作数噪声） |  |
| 121 | 1 | 4 | block4 | Block4：softmax mask 后 rescale 噪声 (`softmax_out_mask_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 122 | 1 | 4 | block4 | Block4：V mask 后 rescale 噪声 (`v_mask_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 123 | 1 | 4 | block4 | Block4：softmax*V 乘法后 rescale 噪声 (`softmax_v_matmul_rescale_sf`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 124 | 1 | 4 | block4 | Block4：softmax*V mask 后 rescale 噪声 (`softmax_v_mask_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 125 | 1 | 4 | block4 | Block4：At*Wo 后 rescale 噪声 (`wo_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 126 | 1 | 4 | block4 | Block4：LN 均值分支 rescale 噪声 (`ln_mean_rescale_sf`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 127 | 1 | 4 | block4 | Block4：(X-u)^2 后 rescale 噪声 (`ln_square_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 128 | 1 | 4 | block4 | Block4：LN 方差分支 rescale 噪声 (`ln_var_rescale_sf`) | SF=28 | True | 28 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 129 | 1 | 4 | block4 | Block4：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 130 | 1 | 5 | block5_n1 | Block5：FFN LN inv_std fresh 噪声 (`inv_std_fresh_sf`) | SF=30 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 131 | 1 | 5 | block5_n1 | Block5：FFN LN x_mean/X_centered fresh 噪声 (`x_centered_fresh_sf`) | SF=30 | True | 30 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 132 | 1 | 5 | block5_n1 | Block5：FFN LN gamma 操作数 encode 噪声 (`gamma_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 133 | 1 | 5 | block5_n1 | Block5：Wffn1 操作数 encode 噪声 (`wffn1_sf`) | SF=22 | True | 22 | 来自 static_skeletons：encode（操作数噪声） |  |
| 134 | 1 | 5 | block5_n1 | Block5：GELU 系数操作数 encode 噪声 (`gelu_coeff_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 135 | 1 | 5 | block5_n1 | Block5：(X-u)/std 后 rescale 噪声 (`normalize_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 136 | 1 | 5 | block5_n1 | Block5：乘 gamma 后 rescale 噪声 (`gamma_rescale_sf`) | SF=30 | True | 30 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 137 | 1 | 5 | block5_n1 | Block5：Wffn1*X 后 rescale 噪声 (`wffn1_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 138 | 1 | 5 | block5_n1 | Block5：GELU x^2 后 rescale 噪声 (`gelu_power_rescale_sf_0`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this power-rescale slot |
| 139 | 1 | 5 | block5_n1 | Block5：GELU x^3 后 rescale 噪声 (`gelu_power_rescale_sf_1`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this power-rescale slot |
| 140 | 1 | 5 | block5_n1 | Block5：GELU x^4 后 rescale 噪声 (`gelu_power_rescale_sf_2`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this power-rescale slot |
| 141 | 1 | 5 | block5_n1 | Block5：GELU degree=1 系数乘法后 rescale 噪声 (`gelu_coeff_mul_rescale_sf_0`) | SF=30 | True | 30 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 142 | 1 | 5 | block5_n1 | Block5：GELU degree=2 系数乘法后 rescale 噪声 (`gelu_coeff_mul_rescale_sf_1`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this coefficient-rescale slot |
| 143 | 1 | 5 | block5_n1 | Block5：GELU degree=3 占位系数 rescale 噪声 (`gelu_coeff_mul_rescale_sf_2`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this coefficient-rescale slot |
| 144 | 1 | 5 | block5_n1 | Block5：GELU degree=4 系数乘法后 rescale 噪声 (`gelu_coeff_mul_rescale_sf_3`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this coefficient-rescale slot |
| 145 | 1 | 5 | block5_n1 | Block5：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 146 | 2 | 1 | block1_mrpc | Block1：GELU 输出 fresh 噪声 (`gelu_out_sf`) | SF=30 | True | 30 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 147 | 2 | 1 | block1_mrpc | Block1：Wffn2 操作数 encode 噪声 (`wffn2_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 148 | 2 | 1 | block1_mrpc | Block1：第一个 1/d 操作数 encode 噪声 (`mean_inv_d_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 149 | 2 | 1 | block1_mrpc | Block1：第二个 1/d 操作数 encode 噪声 (`var_inv_d_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 150 | 2 | 1 | block1_mrpc | Block1：Wffn2*X 后结果 rescale 噪声 (`wffn2_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 151 | 2 | 1 | block1_mrpc | Block1：第一个 1/d 乘法后结果 rescale 噪声 (`mean_rescale_sf`) | SF=34 | True | 34 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 152 | 2 | 1 | block1_mrpc | Block1：(X-u)^2 后结果 rescale 噪声 (`square_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 153 | 2 | 1 | block1_mrpc | Block1：第二个 1/d 乘法后结果 rescale 噪声 (`var_rescale_sf`) | SF=34 | True | 34 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 154 | 2 | 1 | block1_mrpc | Block1：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 155 | 2 | 2 | block2_mrpc | Block2：inv_std/std 相关输入 fresh 噪声 (`inv_std_fresh_sf`) | SF=31 | True | 31 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 156 | 2 | 2 | block2_mrpc | Block2：X_centered 输入 fresh 噪声 (`x_centered_fresh_sf`) | SF=30 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 157 | 2 | 2 | block2_mrpc | Block2：LayerNorm gamma 操作数 encode 噪声 (`gamma_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 158 | 2 | 2 | block2_mrpc | Block2：Wq 操作数 encode 噪声（RO ctpt_wq_wk 共享） (`wq_sf`) | SF=22 | True | 22 | 来自 static_skeletons：encode（操作数噪声） |  |
| 159 | 2 | 2 | block2_mrpc | Block2：Wk 操作数 encode 噪声（RO ctpt_wq_wk 共享） (`wk_sf`) | SF=22 | True | 22 | 来自 static_skeletons：encode（操作数噪声） |  |
| 160 | 2 | 2 | block2_mrpc | Block2：Wv 操作数 encode 噪声 (`wv_sf`) | SF=22 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 161 | 2 | 2 | block2_mrpc | Block2：KT 第一个 mask 操作数 encode 噪声 (`kt_mask1_sf`) | SF=15 | True | 15 | 来自 static_skeletons：encode（操作数噪声） |  |
| 162 | 2 | 2 | block2_mrpc | Block2：KT 第二个 mask 操作数 encode 噪声 (`kt_mask2_sf`) | SF=15 | True | 15 | 来自 static_skeletons：encode（操作数噪声） |  |
| 163 | 2 | 2 | block2_mrpc | Block2：Q 第一个 mask 操作数 encode 噪声 (`q_mask1_sf`) | SF=15 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 164 | 2 | 2 | block2_mrpc | Block2：Q 第二个 mask 操作数 encode 噪声 (`q_mask2_sf`) | SF=15 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 165 | 2 | 2 | block2_mrpc | Block2：QKT merge mask 操作数 encode 噪声 (`qkt_merge_mask_sf`) | SF=15 | True | 15 | 来自 static_skeletons：encode（操作数噪声） |  |
| 166 | 2 | 2 | block2_mrpc | Block2：(X-u)/std 归一化后 rescale 噪声 (`normalize_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 167 | 2 | 2 | block2_mrpc | Block2：乘 gamma 后 rescale 噪声 (`gamma_rescale_sf`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 168 | 2 | 2 | block2_mrpc | Block2：WkX 后 rescale 噪声 (`wk_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 169 | 2 | 2 | block2_mrpc | Block2：WqX 后 rescale 噪声 (`wq_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 170 | 2 | 2 | block2_mrpc | Block2：WvX 后 rescale 噪声 (`wv_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 171 | 2 | 2 | block2_mrpc | Block2：KT 第一个 mask 后 rescale 噪声 (`kt_mask1_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 172 | 2 | 2 | block2_mrpc | Block2：KT 第二个 mask 后 rescale 噪声 (`kt_mask2_rescale_sf`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 173 | 2 | 2 | block2_mrpc | Block2：Q 第一个 mask 后 rescale 噪声 (`q_mask1_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 174 | 2 | 2 | block2_mrpc | Block2：Q 第二个 mask 后 rescale 噪声 (`q_mask2_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 175 | 2 | 2 | block2_mrpc | Block2：Q*KT 后 rescale 噪声 (`qkt_matmul_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 176 | 2 | 2 | block2_mrpc | Block2：QKT merge mask 后 rescale 噪声 (`qkt_merge_mask_rescale_sf`) | SF=28 | True | 28 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 177 | 2 | 2 | block2_mrpc | Block2：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 178 | 2 | 3 | block3_exp_n5 | Block3：softmax 输入 X fresh 噪声 (`x_fresh_sf`) | SF=28 | True | 28 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 179 | 2 | 3 | block3_exp_n5 | Block3：1/(2n) 操作数 encode 噪声 (`inv_2n_sf`) | SF=15 | True | 15 | 来自 static_skeletons：encode（操作数噪声） |  |
| 180 | 2 | 3 | block3_exp_n5 | Block3：X*(1/2n) 后 rescale 噪声 (`x_inv_2n_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 181 | 2 | 3 | block3_exp_n5 | Block3：softmax 平方链第 1 个 rescale 噪声 (`square_rescale_sf_0`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 182 | 2 | 3 | block3_exp_n5 | Block3：softmax 平方链第 2 个 rescale 噪声 (`square_rescale_sf_1`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 183 | 2 | 3 | block3_exp_n5 | Block3：softmax 平方链第 3 个 rescale 噪声 (`square_rescale_sf_2`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 184 | 2 | 3 | block3_exp_n5 | Block3：softmax 平方链第 4 个 rescale 噪声 (`square_rescale_sf_3`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 185 | 2 | 3 | block3_exp_n5 | Block3：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 186 | 2 | 4 | block4 | Block4：softmax 输出 fresh 噪声 (`softmax_out_fresh_sf`) | SF=35 | True | 35 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 187 | 2 | 4 | block4 | Block4：V 输入 fresh 噪声 (`v_fresh_sf`) | SF=30 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 188 | 2 | 4 | block4 | Block4：softmax mask 操作数 encode 噪声 (`softmax_out_mask_sf`) | SF=14 | True | 14 | 来自 static_skeletons：encode（操作数噪声） |  |
| 189 | 2 | 4 | block4 | Block4：V mask 操作数 encode 噪声 (`v_mask_sf`) | SF=14 | True | 14 | 来自 static_skeletons：encode（操作数噪声） |  |
| 190 | 2 | 4 | block4 | Block4：softmax*V mask 操作数 encode 噪声 (`softmax_v_mask_sf`) | SF=14 | True | 14 | 来自 static_skeletons：encode（操作数噪声） |  |
| 191 | 2 | 4 | block4 | Block4：attention LN 第一个 1/d encode 噪声 (`ln_mean_inv_d_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 192 | 2 | 4 | block4 | Block4：attention LN 第二个 1/d encode 噪声 (`ln_var_inv_d_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 193 | 2 | 4 | block4 | Block4：Wo 操作数 encode 噪声 (`wo_sf`) | SF=22 | True | 22 | 来自 static_skeletons：encode（操作数噪声） |  |
| 194 | 2 | 4 | block4 | Block4：softmax mask 后 rescale 噪声 (`softmax_out_mask_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 195 | 2 | 4 | block4 | Block4：V mask 后 rescale 噪声 (`v_mask_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 196 | 2 | 4 | block4 | Block4：softmax*V 乘法后 rescale 噪声 (`softmax_v_matmul_rescale_sf`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 197 | 2 | 4 | block4 | Block4：softmax*V mask 后 rescale 噪声 (`softmax_v_mask_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 198 | 2 | 4 | block4 | Block4：At*Wo 后 rescale 噪声 (`wo_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 199 | 2 | 4 | block4 | Block4：LN 均值分支 rescale 噪声 (`ln_mean_rescale_sf`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 200 | 2 | 4 | block4 | Block4：(X-u)^2 后 rescale 噪声 (`ln_square_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 201 | 2 | 4 | block4 | Block4：LN 方差分支 rescale 噪声 (`ln_var_rescale_sf`) | SF=28 | True | 28 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 202 | 2 | 4 | block4 | Block4：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 203 | 2 | 5 | block5_n1 | Block5：FFN LN inv_std fresh 噪声 (`inv_std_fresh_sf`) | SF=30 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 204 | 2 | 5 | block5_n1 | Block5：FFN LN x_mean/X_centered fresh 噪声 (`x_centered_fresh_sf`) | SF=30 | True | 30 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 205 | 2 | 5 | block5_n1 | Block5：FFN LN gamma 操作数 encode 噪声 (`gamma_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 206 | 2 | 5 | block5_n1 | Block5：Wffn1 操作数 encode 噪声 (`wffn1_sf`) | SF=22 | True | 22 | 来自 static_skeletons：encode（操作数噪声） |  |
| 207 | 2 | 5 | block5_n1 | Block5：GELU 系数操作数 encode 噪声 (`gelu_coeff_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 208 | 2 | 5 | block5_n1 | Block5：(X-u)/std 后 rescale 噪声 (`normalize_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 209 | 2 | 5 | block5_n1 | Block5：乘 gamma 后 rescale 噪声 (`gamma_rescale_sf`) | SF=30 | True | 30 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 210 | 2 | 5 | block5_n1 | Block5：Wffn1*X 后 rescale 噪声 (`wffn1_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 211 | 2 | 5 | block5_n1 | Block5：GELU x^2 后 rescale 噪声 (`gelu_power_rescale_sf_0`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this power-rescale slot |
| 212 | 2 | 5 | block5_n1 | Block5：GELU x^3 后 rescale 噪声 (`gelu_power_rescale_sf_1`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this power-rescale slot |
| 213 | 2 | 5 | block5_n1 | Block5：GELU x^4 后 rescale 噪声 (`gelu_power_rescale_sf_2`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this power-rescale slot |
| 214 | 2 | 5 | block5_n1 | Block5：GELU degree=1 系数乘法后 rescale 噪声 (`gelu_coeff_mul_rescale_sf_0`) | SF=30 | True | 30 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 215 | 2 | 5 | block5_n1 | Block5：GELU degree=2 系数乘法后 rescale 噪声 (`gelu_coeff_mul_rescale_sf_1`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this coefficient-rescale slot |
| 216 | 2 | 5 | block5_n1 | Block5：GELU degree=3 占位系数 rescale 噪声 (`gelu_coeff_mul_rescale_sf_2`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this coefficient-rescale slot |
| 217 | 2 | 5 | block5_n1 | Block5：GELU degree=4 系数乘法后 rescale 噪声 (`gelu_coeff_mul_rescale_sf_3`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this coefficient-rescale slot |
| 218 | 2 | 5 | block5_n1 | Block5：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 219 | 3 | 1 | block1_mrpc | Block1：GELU 输出 fresh 噪声 (`gelu_out_sf`) | SF=30 | True | 30 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 220 | 3 | 1 | block1_mrpc | Block1：Wffn2 操作数 encode 噪声 (`wffn2_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 221 | 3 | 1 | block1_mrpc | Block1：第一个 1/d 操作数 encode 噪声 (`mean_inv_d_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 222 | 3 | 1 | block1_mrpc | Block1：第二个 1/d 操作数 encode 噪声 (`var_inv_d_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 223 | 3 | 1 | block1_mrpc | Block1：Wffn2*X 后结果 rescale 噪声 (`wffn2_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 224 | 3 | 1 | block1_mrpc | Block1：第一个 1/d 乘法后结果 rescale 噪声 (`mean_rescale_sf`) | SF=34 | True | 34 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 225 | 3 | 1 | block1_mrpc | Block1：(X-u)^2 后结果 rescale 噪声 (`square_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 226 | 3 | 1 | block1_mrpc | Block1：第二个 1/d 乘法后结果 rescale 噪声 (`var_rescale_sf`) | SF=34 | True | 34 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 227 | 3 | 1 | block1_mrpc | Block1：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 228 | 3 | 2 | block2_mrpc | Block2：inv_std/std 相关输入 fresh 噪声 (`inv_std_fresh_sf`) | SF=31 | True | 31 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 229 | 3 | 2 | block2_mrpc | Block2：X_centered 输入 fresh 噪声 (`x_centered_fresh_sf`) | SF=30 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 230 | 3 | 2 | block2_mrpc | Block2：LayerNorm gamma 操作数 encode 噪声 (`gamma_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 231 | 3 | 2 | block2_mrpc | Block2：Wq 操作数 encode 噪声（RO ctpt_wq_wk 共享） (`wq_sf`) | SF=22 | True | 22 | 来自 static_skeletons：encode（操作数噪声） |  |
| 232 | 3 | 2 | block2_mrpc | Block2：Wk 操作数 encode 噪声（RO ctpt_wq_wk 共享） (`wk_sf`) | SF=22 | True | 22 | 来自 static_skeletons：encode（操作数噪声） |  |
| 233 | 3 | 2 | block2_mrpc | Block2：Wv 操作数 encode 噪声 (`wv_sf`) | SF=22 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 234 | 3 | 2 | block2_mrpc | Block2：KT 第一个 mask 操作数 encode 噪声 (`kt_mask1_sf`) | SF=15 | True | 15 | 来自 static_skeletons：encode（操作数噪声） |  |
| 235 | 3 | 2 | block2_mrpc | Block2：KT 第二个 mask 操作数 encode 噪声 (`kt_mask2_sf`) | SF=15 | True | 15 | 来自 static_skeletons：encode（操作数噪声） |  |
| 236 | 3 | 2 | block2_mrpc | Block2：Q 第一个 mask 操作数 encode 噪声 (`q_mask1_sf`) | SF=15 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 237 | 3 | 2 | block2_mrpc | Block2：Q 第二个 mask 操作数 encode 噪声 (`q_mask2_sf`) | SF=15 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 238 | 3 | 2 | block2_mrpc | Block2：QKT merge mask 操作数 encode 噪声 (`qkt_merge_mask_sf`) | SF=15 | True | 15 | 来自 static_skeletons：encode（操作数噪声） |  |
| 239 | 3 | 2 | block2_mrpc | Block2：(X-u)/std 归一化后 rescale 噪声 (`normalize_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 240 | 3 | 2 | block2_mrpc | Block2：乘 gamma 后 rescale 噪声 (`gamma_rescale_sf`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 241 | 3 | 2 | block2_mrpc | Block2：WkX 后 rescale 噪声 (`wk_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 242 | 3 | 2 | block2_mrpc | Block2：WqX 后 rescale 噪声 (`wq_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 243 | 3 | 2 | block2_mrpc | Block2：WvX 后 rescale 噪声 (`wv_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 244 | 3 | 2 | block2_mrpc | Block2：KT 第一个 mask 后 rescale 噪声 (`kt_mask1_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 245 | 3 | 2 | block2_mrpc | Block2：KT 第二个 mask 后 rescale 噪声 (`kt_mask2_rescale_sf`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 246 | 3 | 2 | block2_mrpc | Block2：Q 第一个 mask 后 rescale 噪声 (`q_mask1_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 247 | 3 | 2 | block2_mrpc | Block2：Q 第二个 mask 后 rescale 噪声 (`q_mask2_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 248 | 3 | 2 | block2_mrpc | Block2：Q*KT 后 rescale 噪声 (`qkt_matmul_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 249 | 3 | 2 | block2_mrpc | Block2：QKT merge mask 后 rescale 噪声 (`qkt_merge_mask_rescale_sf`) | SF=28 | True | 28 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 250 | 3 | 2 | block2_mrpc | Block2：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 251 | 3 | 3 | block3_exp_n5 | Block3：softmax 输入 X fresh 噪声 (`x_fresh_sf`) | SF=28 | True | 28 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 252 | 3 | 3 | block3_exp_n5 | Block3：1/(2n) 操作数 encode 噪声 (`inv_2n_sf`) | SF=15 | True | 15 | 来自 static_skeletons：encode（操作数噪声） |  |
| 253 | 3 | 3 | block3_exp_n5 | Block3：X*(1/2n) 后 rescale 噪声 (`x_inv_2n_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 254 | 3 | 3 | block3_exp_n5 | Block3：softmax 平方链第 1 个 rescale 噪声 (`square_rescale_sf_0`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 255 | 3 | 3 | block3_exp_n5 | Block3：softmax 平方链第 2 个 rescale 噪声 (`square_rescale_sf_1`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 256 | 3 | 3 | block3_exp_n5 | Block3：softmax 平方链第 3 个 rescale 噪声 (`square_rescale_sf_2`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 257 | 3 | 3 | block3_exp_n5 | Block3：softmax 平方链第 4 个 rescale 噪声 (`square_rescale_sf_3`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 258 | 3 | 3 | block3_exp_n5 | Block3：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 259 | 3 | 4 | block4 | Block4：softmax 输出 fresh 噪声 (`softmax_out_fresh_sf`) | SF=35 | True | 35 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 260 | 3 | 4 | block4 | Block4：V 输入 fresh 噪声 (`v_fresh_sf`) | SF=30 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 261 | 3 | 4 | block4 | Block4：softmax mask 操作数 encode 噪声 (`softmax_out_mask_sf`) | SF=14 | True | 14 | 来自 static_skeletons：encode（操作数噪声） |  |
| 262 | 3 | 4 | block4 | Block4：V mask 操作数 encode 噪声 (`v_mask_sf`) | SF=14 | True | 14 | 来自 static_skeletons：encode（操作数噪声） |  |
| 263 | 3 | 4 | block4 | Block4：softmax*V mask 操作数 encode 噪声 (`softmax_v_mask_sf`) | SF=14 | True | 14 | 来自 static_skeletons：encode（操作数噪声） |  |
| 264 | 3 | 4 | block4 | Block4：attention LN 第一个 1/d encode 噪声 (`ln_mean_inv_d_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 265 | 3 | 4 | block4 | Block4：attention LN 第二个 1/d encode 噪声 (`ln_var_inv_d_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 266 | 3 | 4 | block4 | Block4：Wo 操作数 encode 噪声 (`wo_sf`) | SF=22 | True | 22 | 来自 static_skeletons：encode（操作数噪声） |  |
| 267 | 3 | 4 | block4 | Block4：softmax mask 后 rescale 噪声 (`softmax_out_mask_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 268 | 3 | 4 | block4 | Block4：V mask 后 rescale 噪声 (`v_mask_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 269 | 3 | 4 | block4 | Block4：softmax*V 乘法后 rescale 噪声 (`softmax_v_matmul_rescale_sf`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 270 | 3 | 4 | block4 | Block4：softmax*V mask 后 rescale 噪声 (`softmax_v_mask_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 271 | 3 | 4 | block4 | Block4：At*Wo 后 rescale 噪声 (`wo_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 272 | 3 | 4 | block4 | Block4：LN 均值分支 rescale 噪声 (`ln_mean_rescale_sf`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 273 | 3 | 4 | block4 | Block4：(X-u)^2 后 rescale 噪声 (`ln_square_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 274 | 3 | 4 | block4 | Block4：LN 方差分支 rescale 噪声 (`ln_var_rescale_sf`) | SF=28 | True | 28 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 275 | 3 | 4 | block4 | Block4：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 276 | 3 | 5 | block5_n1 | Block5：FFN LN inv_std fresh 噪声 (`inv_std_fresh_sf`) | SF=30 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 277 | 3 | 5 | block5_n1 | Block5：FFN LN x_mean/X_centered fresh 噪声 (`x_centered_fresh_sf`) | SF=30 | True | 30 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 278 | 3 | 5 | block5_n1 | Block5：FFN LN gamma 操作数 encode 噪声 (`gamma_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 279 | 3 | 5 | block5_n1 | Block5：Wffn1 操作数 encode 噪声 (`wffn1_sf`) | SF=22 | True | 22 | 来自 static_skeletons：encode（操作数噪声） |  |
| 280 | 3 | 5 | block5_n1 | Block5：GELU 系数操作数 encode 噪声 (`gelu_coeff_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 281 | 3 | 5 | block5_n1 | Block5：(X-u)/std 后 rescale 噪声 (`normalize_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 282 | 3 | 5 | block5_n1 | Block5：乘 gamma 后 rescale 噪声 (`gamma_rescale_sf`) | SF=30 | True | 30 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 283 | 3 | 5 | block5_n1 | Block5：Wffn1*X 后 rescale 噪声 (`wffn1_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 284 | 3 | 5 | block5_n1 | Block5：GELU x^2 后 rescale 噪声 (`gelu_power_rescale_sf_0`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this power-rescale slot |
| 285 | 3 | 5 | block5_n1 | Block5：GELU x^3 后 rescale 噪声 (`gelu_power_rescale_sf_1`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this power-rescale slot |
| 286 | 3 | 5 | block5_n1 | Block5：GELU x^4 后 rescale 噪声 (`gelu_power_rescale_sf_2`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this power-rescale slot |
| 287 | 3 | 5 | block5_n1 | Block5：GELU degree=1 系数乘法后 rescale 噪声 (`gelu_coeff_mul_rescale_sf_0`) | SF=30 | True | 30 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 288 | 3 | 5 | block5_n1 | Block5：GELU degree=2 系数乘法后 rescale 噪声 (`gelu_coeff_mul_rescale_sf_1`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this coefficient-rescale slot |
| 289 | 3 | 5 | block5_n1 | Block5：GELU degree=3 占位系数 rescale 噪声 (`gelu_coeff_mul_rescale_sf_2`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this coefficient-rescale slot |
| 290 | 3 | 5 | block5_n1 | Block5：GELU degree=4 系数乘法后 rescale 噪声 (`gelu_coeff_mul_rescale_sf_3`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this coefficient-rescale slot |
| 291 | 3 | 5 | block5_n1 | Block5：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 292 | 4 | 1 | block1_mrpc | Block1：GELU 输出 fresh 噪声 (`gelu_out_sf`) | SF=30 | True | 30 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 293 | 4 | 1 | block1_mrpc | Block1：Wffn2 操作数 encode 噪声 (`wffn2_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 294 | 4 | 1 | block1_mrpc | Block1：第一个 1/d 操作数 encode 噪声 (`mean_inv_d_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 295 | 4 | 1 | block1_mrpc | Block1：第二个 1/d 操作数 encode 噪声 (`var_inv_d_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 296 | 4 | 1 | block1_mrpc | Block1：Wffn2*X 后结果 rescale 噪声 (`wffn2_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 297 | 4 | 1 | block1_mrpc | Block1：第一个 1/d 乘法后结果 rescale 噪声 (`mean_rescale_sf`) | SF=34 | True | 34 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 298 | 4 | 1 | block1_mrpc | Block1：(X-u)^2 后结果 rescale 噪声 (`square_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 299 | 4 | 1 | block1_mrpc | Block1：第二个 1/d 乘法后结果 rescale 噪声 (`var_rescale_sf`) | SF=34 | True | 34 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 300 | 4 | 1 | block1_mrpc | Block1：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 301 | 4 | 2 | block2_mrpc | Block2：inv_std/std 相关输入 fresh 噪声 (`inv_std_fresh_sf`) | SF=31 | True | 31 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 302 | 4 | 2 | block2_mrpc | Block2：X_centered 输入 fresh 噪声 (`x_centered_fresh_sf`) | SF=30 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 303 | 4 | 2 | block2_mrpc | Block2：LayerNorm gamma 操作数 encode 噪声 (`gamma_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 304 | 4 | 2 | block2_mrpc | Block2：Wq 操作数 encode 噪声（RO ctpt_wq_wk 共享） (`wq_sf`) | SF=22 | True | 22 | 来自 static_skeletons：encode（操作数噪声） |  |
| 305 | 4 | 2 | block2_mrpc | Block2：Wk 操作数 encode 噪声（RO ctpt_wq_wk 共享） (`wk_sf`) | SF=22 | True | 22 | 来自 static_skeletons：encode（操作数噪声） |  |
| 306 | 4 | 2 | block2_mrpc | Block2：Wv 操作数 encode 噪声 (`wv_sf`) | SF=22 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 307 | 4 | 2 | block2_mrpc | Block2：KT 第一个 mask 操作数 encode 噪声 (`kt_mask1_sf`) | SF=15 | True | 15 | 来自 static_skeletons：encode（操作数噪声） |  |
| 308 | 4 | 2 | block2_mrpc | Block2：KT 第二个 mask 操作数 encode 噪声 (`kt_mask2_sf`) | SF=15 | True | 15 | 来自 static_skeletons：encode（操作数噪声） |  |
| 309 | 4 | 2 | block2_mrpc | Block2：Q 第一个 mask 操作数 encode 噪声 (`q_mask1_sf`) | SF=15 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 310 | 4 | 2 | block2_mrpc | Block2：Q 第二个 mask 操作数 encode 噪声 (`q_mask2_sf`) | SF=15 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 311 | 4 | 2 | block2_mrpc | Block2：QKT merge mask 操作数 encode 噪声 (`qkt_merge_mask_sf`) | SF=15 | True | 15 | 来自 static_skeletons：encode（操作数噪声） |  |
| 312 | 4 | 2 | block2_mrpc | Block2：(X-u)/std 归一化后 rescale 噪声 (`normalize_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 313 | 4 | 2 | block2_mrpc | Block2：乘 gamma 后 rescale 噪声 (`gamma_rescale_sf`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 314 | 4 | 2 | block2_mrpc | Block2：WkX 后 rescale 噪声 (`wk_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 315 | 4 | 2 | block2_mrpc | Block2：WqX 后 rescale 噪声 (`wq_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 316 | 4 | 2 | block2_mrpc | Block2：WvX 后 rescale 噪声 (`wv_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 317 | 4 | 2 | block2_mrpc | Block2：KT 第一个 mask 后 rescale 噪声 (`kt_mask1_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 318 | 4 | 2 | block2_mrpc | Block2：KT 第二个 mask 后 rescale 噪声 (`kt_mask2_rescale_sf`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 319 | 4 | 2 | block2_mrpc | Block2：Q 第一个 mask 后 rescale 噪声 (`q_mask1_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 320 | 4 | 2 | block2_mrpc | Block2：Q 第二个 mask 后 rescale 噪声 (`q_mask2_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 321 | 4 | 2 | block2_mrpc | Block2：Q*KT 后 rescale 噪声 (`qkt_matmul_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 322 | 4 | 2 | block2_mrpc | Block2：QKT merge mask 后 rescale 噪声 (`qkt_merge_mask_rescale_sf`) | SF=28 | True | 28 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 323 | 4 | 2 | block2_mrpc | Block2：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 324 | 4 | 3 | block3_exp_n5 | Block3：softmax 输入 X fresh 噪声 (`x_fresh_sf`) | SF=28 | True | 28 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 325 | 4 | 3 | block3_exp_n5 | Block3：1/(2n) 操作数 encode 噪声 (`inv_2n_sf`) | SF=15 | True | 15 | 来自 static_skeletons：encode（操作数噪声） |  |
| 326 | 4 | 3 | block3_exp_n5 | Block3：X*(1/2n) 后 rescale 噪声 (`x_inv_2n_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 327 | 4 | 3 | block3_exp_n5 | Block3：softmax 平方链第 1 个 rescale 噪声 (`square_rescale_sf_0`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 328 | 4 | 3 | block3_exp_n5 | Block3：softmax 平方链第 2 个 rescale 噪声 (`square_rescale_sf_1`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 329 | 4 | 3 | block3_exp_n5 | Block3：softmax 平方链第 3 个 rescale 噪声 (`square_rescale_sf_2`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 330 | 4 | 3 | block3_exp_n5 | Block3：softmax 平方链第 4 个 rescale 噪声 (`square_rescale_sf_3`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 331 | 4 | 3 | block3_exp_n5 | Block3：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 332 | 4 | 4 | block4 | Block4：softmax 输出 fresh 噪声 (`softmax_out_fresh_sf`) | SF=35 | True | 35 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 333 | 4 | 4 | block4 | Block4：V 输入 fresh 噪声 (`v_fresh_sf`) | SF=30 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 334 | 4 | 4 | block4 | Block4：softmax mask 操作数 encode 噪声 (`softmax_out_mask_sf`) | SF=14 | True | 14 | 来自 static_skeletons：encode（操作数噪声） |  |
| 335 | 4 | 4 | block4 | Block4：V mask 操作数 encode 噪声 (`v_mask_sf`) | SF=14 | True | 14 | 来自 static_skeletons：encode（操作数噪声） |  |
| 336 | 4 | 4 | block4 | Block4：softmax*V mask 操作数 encode 噪声 (`softmax_v_mask_sf`) | SF=14 | True | 14 | 来自 static_skeletons：encode（操作数噪声） |  |
| 337 | 4 | 4 | block4 | Block4：attention LN 第一个 1/d encode 噪声 (`ln_mean_inv_d_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 338 | 4 | 4 | block4 | Block4：attention LN 第二个 1/d encode 噪声 (`ln_var_inv_d_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 339 | 4 | 4 | block4 | Block4：Wo 操作数 encode 噪声 (`wo_sf`) | SF=22 | True | 22 | 来自 static_skeletons：encode（操作数噪声） |  |
| 340 | 4 | 4 | block4 | Block4：softmax mask 后 rescale 噪声 (`softmax_out_mask_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 341 | 4 | 4 | block4 | Block4：V mask 后 rescale 噪声 (`v_mask_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 342 | 4 | 4 | block4 | Block4：softmax*V 乘法后 rescale 噪声 (`softmax_v_matmul_rescale_sf`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 343 | 4 | 4 | block4 | Block4：softmax*V mask 后 rescale 噪声 (`softmax_v_mask_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 344 | 4 | 4 | block4 | Block4：At*Wo 后 rescale 噪声 (`wo_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 345 | 4 | 4 | block4 | Block4：LN 均值分支 rescale 噪声 (`ln_mean_rescale_sf`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 346 | 4 | 4 | block4 | Block4：(X-u)^2 后 rescale 噪声 (`ln_square_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 347 | 4 | 4 | block4 | Block4：LN 方差分支 rescale 噪声 (`ln_var_rescale_sf`) | SF=28 | True | 28 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 348 | 4 | 4 | block4 | Block4：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 349 | 4 | 5 | block5_n1 | Block5：FFN LN inv_std fresh 噪声 (`inv_std_fresh_sf`) | SF=30 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 350 | 4 | 5 | block5_n1 | Block5：FFN LN x_mean/X_centered fresh 噪声 (`x_centered_fresh_sf`) | SF=30 | True | 30 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 351 | 4 | 5 | block5_n1 | Block5：FFN LN gamma 操作数 encode 噪声 (`gamma_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 352 | 4 | 5 | block5_n1 | Block5：Wffn1 操作数 encode 噪声 (`wffn1_sf`) | SF=22 | True | 22 | 来自 static_skeletons：encode（操作数噪声） |  |
| 353 | 4 | 5 | block5_n1 | Block5：GELU 系数操作数 encode 噪声 (`gelu_coeff_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 354 | 4 | 5 | block5_n1 | Block5：(X-u)/std 后 rescale 噪声 (`normalize_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 355 | 4 | 5 | block5_n1 | Block5：乘 gamma 后 rescale 噪声 (`gamma_rescale_sf`) | SF=30 | True | 30 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 356 | 4 | 5 | block5_n1 | Block5：Wffn1*X 后 rescale 噪声 (`wffn1_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 357 | 4 | 5 | block5_n1 | Block5：GELU x^2 后 rescale 噪声 (`gelu_power_rescale_sf_0`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this power-rescale slot |
| 358 | 4 | 5 | block5_n1 | Block5：GELU x^3 后 rescale 噪声 (`gelu_power_rescale_sf_1`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this power-rescale slot |
| 359 | 4 | 5 | block5_n1 | Block5：GELU x^4 后 rescale 噪声 (`gelu_power_rescale_sf_2`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this power-rescale slot |
| 360 | 4 | 5 | block5_n1 | Block5：GELU degree=1 系数乘法后 rescale 噪声 (`gelu_coeff_mul_rescale_sf_0`) | SF=30 | True | 30 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 361 | 4 | 5 | block5_n1 | Block5：GELU degree=2 系数乘法后 rescale 噪声 (`gelu_coeff_mul_rescale_sf_1`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this coefficient-rescale slot |
| 362 | 4 | 5 | block5_n1 | Block5：GELU degree=3 占位系数 rescale 噪声 (`gelu_coeff_mul_rescale_sf_2`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this coefficient-rescale slot |
| 363 | 4 | 5 | block5_n1 | Block5：GELU degree=4 系数乘法后 rescale 噪声 (`gelu_coeff_mul_rescale_sf_3`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this coefficient-rescale slot |
| 364 | 4 | 5 | block5_n1 | Block5：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 365 | 5 | 1 | block1_mrpc | Block1：GELU 输出 fresh 噪声 (`gelu_out_sf`) | SF=30 | True | 30 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 366 | 5 | 1 | block1_mrpc | Block1：Wffn2 操作数 encode 噪声 (`wffn2_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 367 | 5 | 1 | block1_mrpc | Block1：第一个 1/d 操作数 encode 噪声 (`mean_inv_d_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 368 | 5 | 1 | block1_mrpc | Block1：第二个 1/d 操作数 encode 噪声 (`var_inv_d_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 369 | 5 | 1 | block1_mrpc | Block1：Wffn2*X 后结果 rescale 噪声 (`wffn2_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 370 | 5 | 1 | block1_mrpc | Block1：第一个 1/d 乘法后结果 rescale 噪声 (`mean_rescale_sf`) | SF=34 | True | 34 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 371 | 5 | 1 | block1_mrpc | Block1：(X-u)^2 后结果 rescale 噪声 (`square_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 372 | 5 | 1 | block1_mrpc | Block1：第二个 1/d 乘法后结果 rescale 噪声 (`var_rescale_sf`) | SF=34 | True | 34 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 373 | 5 | 1 | block1_mrpc | Block1：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 374 | 5 | 2 | block2_mrpc | Block2：inv_std/std 相关输入 fresh 噪声 (`inv_std_fresh_sf`) | SF=31 | True | 31 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 375 | 5 | 2 | block2_mrpc | Block2：X_centered 输入 fresh 噪声 (`x_centered_fresh_sf`) | SF=30 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 376 | 5 | 2 | block2_mrpc | Block2：LayerNorm gamma 操作数 encode 噪声 (`gamma_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 377 | 5 | 2 | block2_mrpc | Block2：Wq 操作数 encode 噪声（RO ctpt_wq_wk 共享） (`wq_sf`) | SF=22 | True | 22 | 来自 static_skeletons：encode（操作数噪声） |  |
| 378 | 5 | 2 | block2_mrpc | Block2：Wk 操作数 encode 噪声（RO ctpt_wq_wk 共享） (`wk_sf`) | SF=22 | True | 22 | 来自 static_skeletons：encode（操作数噪声） |  |
| 379 | 5 | 2 | block2_mrpc | Block2：Wv 操作数 encode 噪声 (`wv_sf`) | SF=22 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 380 | 5 | 2 | block2_mrpc | Block2：KT 第一个 mask 操作数 encode 噪声 (`kt_mask1_sf`) | SF=15 | True | 15 | 来自 static_skeletons：encode（操作数噪声） |  |
| 381 | 5 | 2 | block2_mrpc | Block2：KT 第二个 mask 操作数 encode 噪声 (`kt_mask2_sf`) | SF=15 | True | 15 | 来自 static_skeletons：encode（操作数噪声） |  |
| 382 | 5 | 2 | block2_mrpc | Block2：Q 第一个 mask 操作数 encode 噪声 (`q_mask1_sf`) | SF=15 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 383 | 5 | 2 | block2_mrpc | Block2：Q 第二个 mask 操作数 encode 噪声 (`q_mask2_sf`) | SF=15 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 384 | 5 | 2 | block2_mrpc | Block2：QKT merge mask 操作数 encode 噪声 (`qkt_merge_mask_sf`) | SF=15 | True | 15 | 来自 static_skeletons：encode（操作数噪声） |  |
| 385 | 5 | 2 | block2_mrpc | Block2：(X-u)/std 归一化后 rescale 噪声 (`normalize_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 386 | 5 | 2 | block2_mrpc | Block2：乘 gamma 后 rescale 噪声 (`gamma_rescale_sf`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 387 | 5 | 2 | block2_mrpc | Block2：WkX 后 rescale 噪声 (`wk_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 388 | 5 | 2 | block2_mrpc | Block2：WqX 后 rescale 噪声 (`wq_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 389 | 5 | 2 | block2_mrpc | Block2：WvX 后 rescale 噪声 (`wv_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 390 | 5 | 2 | block2_mrpc | Block2：KT 第一个 mask 后 rescale 噪声 (`kt_mask1_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 391 | 5 | 2 | block2_mrpc | Block2：KT 第二个 mask 后 rescale 噪声 (`kt_mask2_rescale_sf`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 392 | 5 | 2 | block2_mrpc | Block2：Q 第一个 mask 后 rescale 噪声 (`q_mask1_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 393 | 5 | 2 | block2_mrpc | Block2：Q 第二个 mask 后 rescale 噪声 (`q_mask2_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 394 | 5 | 2 | block2_mrpc | Block2：Q*KT 后 rescale 噪声 (`qkt_matmul_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 395 | 5 | 2 | block2_mrpc | Block2：QKT merge mask 后 rescale 噪声 (`qkt_merge_mask_rescale_sf`) | SF=28 | True | 28 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 396 | 5 | 2 | block2_mrpc | Block2：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 397 | 5 | 3 | block3_exp_n2 | Block3：softmax 输入 X fresh 噪声 (`x_fresh_sf`) | SF=27 | True | 27 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 398 | 5 | 3 | block3_exp_n2 | Block3：1/(2n) 操作数 encode 噪声 (`inv_2n_sf`) | SF=16 | True | 16 | 来自 static_skeletons：encode（操作数噪声） |  |
| 399 | 5 | 3 | block3_exp_n2 | Block3：X*(1/2n) 后 rescale 噪声 (`x_inv_2n_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 400 | 5 | 3 | block3_exp_n2 | Block3：softmax 平方链第 1 个 rescale 噪声 (`square_rescale_sf_0`) | SF=34 | True | 34 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 401 | 5 | 3 | block3_exp_n2 | Block3：softmax 平方链第 2 个 rescale 噪声 (`square_rescale_sf_1`) | SF=34 | True | 34 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 402 | 5 | 3 | block3_exp_n2 | Block3：softmax 平方链第 3 个 rescale 噪声 (`square_rescale_sf_2`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | softmax degree 2 does not use this square-rescale slot |
| 403 | 5 | 3 | block3_exp_n2 | Block3：softmax 平方链第 4 个 rescale 噪声 (`square_rescale_sf_3`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | softmax degree 2 does not use this square-rescale slot |
| 404 | 5 | 3 | block3_exp_n2 | Block3：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 405 | 5 | 4 | block4 | Block4：softmax 输出 fresh 噪声 (`softmax_out_fresh_sf`) | SF=35 | True | 35 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 406 | 5 | 4 | block4 | Block4：V 输入 fresh 噪声 (`v_fresh_sf`) | SF=30 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 407 | 5 | 4 | block4 | Block4：softmax mask 操作数 encode 噪声 (`softmax_out_mask_sf`) | SF=14 | True | 14 | 来自 static_skeletons：encode（操作数噪声） |  |
| 408 | 5 | 4 | block4 | Block4：V mask 操作数 encode 噪声 (`v_mask_sf`) | SF=14 | True | 14 | 来自 static_skeletons：encode（操作数噪声） |  |
| 409 | 5 | 4 | block4 | Block4：softmax*V mask 操作数 encode 噪声 (`softmax_v_mask_sf`) | SF=14 | True | 14 | 来自 static_skeletons：encode（操作数噪声） |  |
| 410 | 5 | 4 | block4 | Block4：attention LN 第一个 1/d encode 噪声 (`ln_mean_inv_d_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 411 | 5 | 4 | block4 | Block4：attention LN 第二个 1/d encode 噪声 (`ln_var_inv_d_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 412 | 5 | 4 | block4 | Block4：Wo 操作数 encode 噪声 (`wo_sf`) | SF=22 | True | 22 | 来自 static_skeletons：encode（操作数噪声） |  |
| 413 | 5 | 4 | block4 | Block4：softmax mask 后 rescale 噪声 (`softmax_out_mask_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 414 | 5 | 4 | block4 | Block4：V mask 后 rescale 噪声 (`v_mask_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 415 | 5 | 4 | block4 | Block4：softmax*V 乘法后 rescale 噪声 (`softmax_v_matmul_rescale_sf`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 416 | 5 | 4 | block4 | Block4：softmax*V mask 后 rescale 噪声 (`softmax_v_mask_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 417 | 5 | 4 | block4 | Block4：At*Wo 后 rescale 噪声 (`wo_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 418 | 5 | 4 | block4 | Block4：LN 均值分支 rescale 噪声 (`ln_mean_rescale_sf`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 419 | 5 | 4 | block4 | Block4：(X-u)^2 后 rescale 噪声 (`ln_square_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 420 | 5 | 4 | block4 | Block4：LN 方差分支 rescale 噪声 (`ln_var_rescale_sf`) | SF=28 | True | 28 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 421 | 5 | 4 | block4 | Block4：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 422 | 5 | 5 | block5_n4 | Block5：FFN LN inv_std fresh 噪声 (`inv_std_fresh_sf`) | SF=30 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 423 | 5 | 5 | block5_n4 | Block5：FFN LN x_mean/X_centered fresh 噪声 (`x_centered_fresh_sf`) | SF=31 | True | 31 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 424 | 5 | 5 | block5_n4 | Block5：FFN LN gamma 操作数 encode 噪声 (`gamma_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 425 | 5 | 5 | block5_n4 | Block5：Wffn1 操作数 encode 噪声 (`wffn1_sf`) | SF=22 | True | 22 | 来自 static_skeletons：encode（操作数噪声） |  |
| 426 | 5 | 5 | block5_n4 | Block5：GELU 系数操作数 encode 噪声 (`gelu_coeff_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 427 | 5 | 5 | block5_n4 | Block5：(X-u)/std 后 rescale 噪声 (`normalize_rescale_sf`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 428 | 5 | 5 | block5_n4 | Block5：乘 gamma 后 rescale 噪声 (`gamma_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 429 | 5 | 5 | block5_n4 | Block5：Wffn1*X 后 rescale 噪声 (`wffn1_rescale_sf`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 430 | 5 | 5 | block5_n4 | Block5：GELU x^2 后 rescale 噪声 (`gelu_power_rescale_sf_0`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 431 | 5 | 5 | block5_n4 | Block5：GELU x^3 后 rescale 噪声 (`gelu_power_rescale_sf_1`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 432 | 5 | 5 | block5_n4 | Block5：GELU x^4 后 rescale 噪声 (`gelu_power_rescale_sf_2`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 433 | 5 | 5 | block5_n4 | Block5：GELU degree=1 系数乘法后 rescale 噪声 (`gelu_coeff_mul_rescale_sf_0`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 434 | 5 | 5 | block5_n4 | Block5：GELU degree=2 系数乘法后 rescale 噪声 (`gelu_coeff_mul_rescale_sf_1`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 435 | 5 | 5 | block5_n4 | Block5：GELU degree=3 占位系数 rescale 噪声 (`gelu_coeff_mul_rescale_sf_2`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 436 | 5 | 5 | block5_n4 | Block5：GELU degree=4 系数乘法后 rescale 噪声 (`gelu_coeff_mul_rescale_sf_3`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 437 | 5 | 5 | block5_n4 | Block5：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 438 | 6 | 1 | block1_mrpc | Block1：GELU 输出 fresh 噪声 (`gelu_out_sf`) | SF=30 | True | 30 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 439 | 6 | 1 | block1_mrpc | Block1：Wffn2 操作数 encode 噪声 (`wffn2_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 440 | 6 | 1 | block1_mrpc | Block1：第一个 1/d 操作数 encode 噪声 (`mean_inv_d_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 441 | 6 | 1 | block1_mrpc | Block1：第二个 1/d 操作数 encode 噪声 (`var_inv_d_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 442 | 6 | 1 | block1_mrpc | Block1：Wffn2*X 后结果 rescale 噪声 (`wffn2_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 443 | 6 | 1 | block1_mrpc | Block1：第一个 1/d 乘法后结果 rescale 噪声 (`mean_rescale_sf`) | SF=34 | True | 34 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 444 | 6 | 1 | block1_mrpc | Block1：(X-u)^2 后结果 rescale 噪声 (`square_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 445 | 6 | 1 | block1_mrpc | Block1：第二个 1/d 乘法后结果 rescale 噪声 (`var_rescale_sf`) | SF=34 | True | 34 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 446 | 6 | 1 | block1_mrpc | Block1：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 447 | 6 | 2 | block2_mrpc | Block2：inv_std/std 相关输入 fresh 噪声 (`inv_std_fresh_sf`) | SF=31 | True | 31 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 448 | 6 | 2 | block2_mrpc | Block2：X_centered 输入 fresh 噪声 (`x_centered_fresh_sf`) | SF=30 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 449 | 6 | 2 | block2_mrpc | Block2：LayerNorm gamma 操作数 encode 噪声 (`gamma_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 450 | 6 | 2 | block2_mrpc | Block2：Wq 操作数 encode 噪声（RO ctpt_wq_wk 共享） (`wq_sf`) | SF=22 | True | 22 | 来自 static_skeletons：encode（操作数噪声） |  |
| 451 | 6 | 2 | block2_mrpc | Block2：Wk 操作数 encode 噪声（RO ctpt_wq_wk 共享） (`wk_sf`) | SF=22 | True | 22 | 来自 static_skeletons：encode（操作数噪声） |  |
| 452 | 6 | 2 | block2_mrpc | Block2：Wv 操作数 encode 噪声 (`wv_sf`) | SF=22 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 453 | 6 | 2 | block2_mrpc | Block2：KT 第一个 mask 操作数 encode 噪声 (`kt_mask1_sf`) | SF=15 | True | 15 | 来自 static_skeletons：encode（操作数噪声） |  |
| 454 | 6 | 2 | block2_mrpc | Block2：KT 第二个 mask 操作数 encode 噪声 (`kt_mask2_sf`) | SF=15 | True | 15 | 来自 static_skeletons：encode（操作数噪声） |  |
| 455 | 6 | 2 | block2_mrpc | Block2：Q 第一个 mask 操作数 encode 噪声 (`q_mask1_sf`) | SF=15 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 456 | 6 | 2 | block2_mrpc | Block2：Q 第二个 mask 操作数 encode 噪声 (`q_mask2_sf`) | SF=15 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 457 | 6 | 2 | block2_mrpc | Block2：QKT merge mask 操作数 encode 噪声 (`qkt_merge_mask_sf`) | SF=15 | True | 15 | 来自 static_skeletons：encode（操作数噪声） |  |
| 458 | 6 | 2 | block2_mrpc | Block2：(X-u)/std 归一化后 rescale 噪声 (`normalize_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 459 | 6 | 2 | block2_mrpc | Block2：乘 gamma 后 rescale 噪声 (`gamma_rescale_sf`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 460 | 6 | 2 | block2_mrpc | Block2：WkX 后 rescale 噪声 (`wk_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 461 | 6 | 2 | block2_mrpc | Block2：WqX 后 rescale 噪声 (`wq_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 462 | 6 | 2 | block2_mrpc | Block2：WvX 后 rescale 噪声 (`wv_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 463 | 6 | 2 | block2_mrpc | Block2：KT 第一个 mask 后 rescale 噪声 (`kt_mask1_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 464 | 6 | 2 | block2_mrpc | Block2：KT 第二个 mask 后 rescale 噪声 (`kt_mask2_rescale_sf`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 465 | 6 | 2 | block2_mrpc | Block2：Q 第一个 mask 后 rescale 噪声 (`q_mask1_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 466 | 6 | 2 | block2_mrpc | Block2：Q 第二个 mask 后 rescale 噪声 (`q_mask2_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 467 | 6 | 2 | block2_mrpc | Block2：Q*KT 后 rescale 噪声 (`qkt_matmul_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 468 | 6 | 2 | block2_mrpc | Block2：QKT merge mask 后 rescale 噪声 (`qkt_merge_mask_rescale_sf`) | SF=28 | True | 28 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 469 | 6 | 2 | block2_mrpc | Block2：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 470 | 6 | 3 | block3_exp_n5 | Block3：softmax 输入 X fresh 噪声 (`x_fresh_sf`) | SF=28 | True | 28 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 471 | 6 | 3 | block3_exp_n5 | Block3：1/(2n) 操作数 encode 噪声 (`inv_2n_sf`) | SF=15 | True | 15 | 来自 static_skeletons：encode（操作数噪声） |  |
| 472 | 6 | 3 | block3_exp_n5 | Block3：X*(1/2n) 后 rescale 噪声 (`x_inv_2n_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 473 | 6 | 3 | block3_exp_n5 | Block3：softmax 平方链第 1 个 rescale 噪声 (`square_rescale_sf_0`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 474 | 6 | 3 | block3_exp_n5 | Block3：softmax 平方链第 2 个 rescale 噪声 (`square_rescale_sf_1`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 475 | 6 | 3 | block3_exp_n5 | Block3：softmax 平方链第 3 个 rescale 噪声 (`square_rescale_sf_2`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 476 | 6 | 3 | block3_exp_n5 | Block3：softmax 平方链第 4 个 rescale 噪声 (`square_rescale_sf_3`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 477 | 6 | 3 | block3_exp_n5 | Block3：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 478 | 6 | 4 | block4 | Block4：softmax 输出 fresh 噪声 (`softmax_out_fresh_sf`) | SF=35 | True | 35 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 479 | 6 | 4 | block4 | Block4：V 输入 fresh 噪声 (`v_fresh_sf`) | SF=30 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 480 | 6 | 4 | block4 | Block4：softmax mask 操作数 encode 噪声 (`softmax_out_mask_sf`) | SF=14 | True | 14 | 来自 static_skeletons：encode（操作数噪声） |  |
| 481 | 6 | 4 | block4 | Block4：V mask 操作数 encode 噪声 (`v_mask_sf`) | SF=14 | True | 14 | 来自 static_skeletons：encode（操作数噪声） |  |
| 482 | 6 | 4 | block4 | Block4：softmax*V mask 操作数 encode 噪声 (`softmax_v_mask_sf`) | SF=14 | True | 14 | 来自 static_skeletons：encode（操作数噪声） |  |
| 483 | 6 | 4 | block4 | Block4：attention LN 第一个 1/d encode 噪声 (`ln_mean_inv_d_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 484 | 6 | 4 | block4 | Block4：attention LN 第二个 1/d encode 噪声 (`ln_var_inv_d_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 485 | 6 | 4 | block4 | Block4：Wo 操作数 encode 噪声 (`wo_sf`) | SF=22 | True | 22 | 来自 static_skeletons：encode（操作数噪声） |  |
| 486 | 6 | 4 | block4 | Block4：softmax mask 后 rescale 噪声 (`softmax_out_mask_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 487 | 6 | 4 | block4 | Block4：V mask 后 rescale 噪声 (`v_mask_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 488 | 6 | 4 | block4 | Block4：softmax*V 乘法后 rescale 噪声 (`softmax_v_matmul_rescale_sf`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 489 | 6 | 4 | block4 | Block4：softmax*V mask 后 rescale 噪声 (`softmax_v_mask_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 490 | 6 | 4 | block4 | Block4：At*Wo 后 rescale 噪声 (`wo_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 491 | 6 | 4 | block4 | Block4：LN 均值分支 rescale 噪声 (`ln_mean_rescale_sf`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 492 | 6 | 4 | block4 | Block4：(X-u)^2 后 rescale 噪声 (`ln_square_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 493 | 6 | 4 | block4 | Block4：LN 方差分支 rescale 噪声 (`ln_var_rescale_sf`) | SF=28 | True | 28 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 494 | 6 | 4 | block4 | Block4：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 495 | 6 | 5 | block5_n1 | Block5：FFN LN inv_std fresh 噪声 (`inv_std_fresh_sf`) | SF=30 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 496 | 6 | 5 | block5_n1 | Block5：FFN LN x_mean/X_centered fresh 噪声 (`x_centered_fresh_sf`) | SF=30 | True | 30 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 497 | 6 | 5 | block5_n1 | Block5：FFN LN gamma 操作数 encode 噪声 (`gamma_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 498 | 6 | 5 | block5_n1 | Block5：Wffn1 操作数 encode 噪声 (`wffn1_sf`) | SF=22 | True | 22 | 来自 static_skeletons：encode（操作数噪声） |  |
| 499 | 6 | 5 | block5_n1 | Block5：GELU 系数操作数 encode 噪声 (`gelu_coeff_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 500 | 6 | 5 | block5_n1 | Block5：(X-u)/std 后 rescale 噪声 (`normalize_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 501 | 6 | 5 | block5_n1 | Block5：乘 gamma 后 rescale 噪声 (`gamma_rescale_sf`) | SF=30 | True | 30 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 502 | 6 | 5 | block5_n1 | Block5：Wffn1*X 后 rescale 噪声 (`wffn1_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 503 | 6 | 5 | block5_n1 | Block5：GELU x^2 后 rescale 噪声 (`gelu_power_rescale_sf_0`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this power-rescale slot |
| 504 | 6 | 5 | block5_n1 | Block5：GELU x^3 后 rescale 噪声 (`gelu_power_rescale_sf_1`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this power-rescale slot |
| 505 | 6 | 5 | block5_n1 | Block5：GELU x^4 后 rescale 噪声 (`gelu_power_rescale_sf_2`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this power-rescale slot |
| 506 | 6 | 5 | block5_n1 | Block5：GELU degree=1 系数乘法后 rescale 噪声 (`gelu_coeff_mul_rescale_sf_0`) | SF=30 | True | 30 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 507 | 6 | 5 | block5_n1 | Block5：GELU degree=2 系数乘法后 rescale 噪声 (`gelu_coeff_mul_rescale_sf_1`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this coefficient-rescale slot |
| 508 | 6 | 5 | block5_n1 | Block5：GELU degree=3 占位系数 rescale 噪声 (`gelu_coeff_mul_rescale_sf_2`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this coefficient-rescale slot |
| 509 | 6 | 5 | block5_n1 | Block5：GELU degree=4 系数乘法后 rescale 噪声 (`gelu_coeff_mul_rescale_sf_3`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this coefficient-rescale slot |
| 510 | 6 | 5 | block5_n1 | Block5：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 511 | 7 | 1 | block1_mrpc | Block1：GELU 输出 fresh 噪声 (`gelu_out_sf`) | SF=30 | True | 30 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 512 | 7 | 1 | block1_mrpc | Block1：Wffn2 操作数 encode 噪声 (`wffn2_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 513 | 7 | 1 | block1_mrpc | Block1：第一个 1/d 操作数 encode 噪声 (`mean_inv_d_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 514 | 7 | 1 | block1_mrpc | Block1：第二个 1/d 操作数 encode 噪声 (`var_inv_d_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 515 | 7 | 1 | block1_mrpc | Block1：Wffn2*X 后结果 rescale 噪声 (`wffn2_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 516 | 7 | 1 | block1_mrpc | Block1：第一个 1/d 乘法后结果 rescale 噪声 (`mean_rescale_sf`) | SF=34 | True | 34 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 517 | 7 | 1 | block1_mrpc | Block1：(X-u)^2 后结果 rescale 噪声 (`square_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 518 | 7 | 1 | block1_mrpc | Block1：第二个 1/d 乘法后结果 rescale 噪声 (`var_rescale_sf`) | SF=34 | True | 34 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 519 | 7 | 1 | block1_mrpc | Block1：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 520 | 7 | 2 | block2_mrpc | Block2：inv_std/std 相关输入 fresh 噪声 (`inv_std_fresh_sf`) | SF=31 | True | 31 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 521 | 7 | 2 | block2_mrpc | Block2：X_centered 输入 fresh 噪声 (`x_centered_fresh_sf`) | SF=30 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 522 | 7 | 2 | block2_mrpc | Block2：LayerNorm gamma 操作数 encode 噪声 (`gamma_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 523 | 7 | 2 | block2_mrpc | Block2：Wq 操作数 encode 噪声（RO ctpt_wq_wk 共享） (`wq_sf`) | SF=22 | True | 22 | 来自 static_skeletons：encode（操作数噪声） |  |
| 524 | 7 | 2 | block2_mrpc | Block2：Wk 操作数 encode 噪声（RO ctpt_wq_wk 共享） (`wk_sf`) | SF=22 | True | 22 | 来自 static_skeletons：encode（操作数噪声） |  |
| 525 | 7 | 2 | block2_mrpc | Block2：Wv 操作数 encode 噪声 (`wv_sf`) | SF=22 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 526 | 7 | 2 | block2_mrpc | Block2：KT 第一个 mask 操作数 encode 噪声 (`kt_mask1_sf`) | SF=15 | True | 15 | 来自 static_skeletons：encode（操作数噪声） |  |
| 527 | 7 | 2 | block2_mrpc | Block2：KT 第二个 mask 操作数 encode 噪声 (`kt_mask2_sf`) | SF=15 | True | 15 | 来自 static_skeletons：encode（操作数噪声） |  |
| 528 | 7 | 2 | block2_mrpc | Block2：Q 第一个 mask 操作数 encode 噪声 (`q_mask1_sf`) | SF=15 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 529 | 7 | 2 | block2_mrpc | Block2：Q 第二个 mask 操作数 encode 噪声 (`q_mask2_sf`) | SF=15 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 530 | 7 | 2 | block2_mrpc | Block2：QKT merge mask 操作数 encode 噪声 (`qkt_merge_mask_sf`) | SF=15 | True | 15 | 来自 static_skeletons：encode（操作数噪声） |  |
| 531 | 7 | 2 | block2_mrpc | Block2：(X-u)/std 归一化后 rescale 噪声 (`normalize_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 532 | 7 | 2 | block2_mrpc | Block2：乘 gamma 后 rescale 噪声 (`gamma_rescale_sf`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 533 | 7 | 2 | block2_mrpc | Block2：WkX 后 rescale 噪声 (`wk_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 534 | 7 | 2 | block2_mrpc | Block2：WqX 后 rescale 噪声 (`wq_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 535 | 7 | 2 | block2_mrpc | Block2：WvX 后 rescale 噪声 (`wv_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 536 | 7 | 2 | block2_mrpc | Block2：KT 第一个 mask 后 rescale 噪声 (`kt_mask1_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 537 | 7 | 2 | block2_mrpc | Block2：KT 第二个 mask 后 rescale 噪声 (`kt_mask2_rescale_sf`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 538 | 7 | 2 | block2_mrpc | Block2：Q 第一个 mask 后 rescale 噪声 (`q_mask1_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 539 | 7 | 2 | block2_mrpc | Block2：Q 第二个 mask 后 rescale 噪声 (`q_mask2_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 540 | 7 | 2 | block2_mrpc | Block2：Q*KT 后 rescale 噪声 (`qkt_matmul_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 541 | 7 | 2 | block2_mrpc | Block2：QKT merge mask 后 rescale 噪声 (`qkt_merge_mask_rescale_sf`) | SF=28 | True | 28 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 542 | 7 | 2 | block2_mrpc | Block2：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 543 | 7 | 3 | block3_exp_n2 | Block3：softmax 输入 X fresh 噪声 (`x_fresh_sf`) | SF=27 | True | 27 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 544 | 7 | 3 | block3_exp_n2 | Block3：1/(2n) 操作数 encode 噪声 (`inv_2n_sf`) | SF=16 | True | 16 | 来自 static_skeletons：encode（操作数噪声） |  |
| 545 | 7 | 3 | block3_exp_n2 | Block3：X*(1/2n) 后 rescale 噪声 (`x_inv_2n_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 546 | 7 | 3 | block3_exp_n2 | Block3：softmax 平方链第 1 个 rescale 噪声 (`square_rescale_sf_0`) | SF=34 | True | 34 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 547 | 7 | 3 | block3_exp_n2 | Block3：softmax 平方链第 2 个 rescale 噪声 (`square_rescale_sf_1`) | SF=34 | True | 34 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 548 | 7 | 3 | block3_exp_n2 | Block3：softmax 平方链第 3 个 rescale 噪声 (`square_rescale_sf_2`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | softmax degree 2 does not use this square-rescale slot |
| 549 | 7 | 3 | block3_exp_n2 | Block3：softmax 平方链第 4 个 rescale 噪声 (`square_rescale_sf_3`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | softmax degree 2 does not use this square-rescale slot |
| 550 | 7 | 3 | block3_exp_n2 | Block3：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 551 | 7 | 4 | block4 | Block4：softmax 输出 fresh 噪声 (`softmax_out_fresh_sf`) | SF=35 | True | 35 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 552 | 7 | 4 | block4 | Block4：V 输入 fresh 噪声 (`v_fresh_sf`) | SF=30 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 553 | 7 | 4 | block4 | Block4：softmax mask 操作数 encode 噪声 (`softmax_out_mask_sf`) | SF=14 | True | 14 | 来自 static_skeletons：encode（操作数噪声） |  |
| 554 | 7 | 4 | block4 | Block4：V mask 操作数 encode 噪声 (`v_mask_sf`) | SF=14 | True | 14 | 来自 static_skeletons：encode（操作数噪声） |  |
| 555 | 7 | 4 | block4 | Block4：softmax*V mask 操作数 encode 噪声 (`softmax_v_mask_sf`) | SF=14 | True | 14 | 来自 static_skeletons：encode（操作数噪声） |  |
| 556 | 7 | 4 | block4 | Block4：attention LN 第一个 1/d encode 噪声 (`ln_mean_inv_d_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 557 | 7 | 4 | block4 | Block4：attention LN 第二个 1/d encode 噪声 (`ln_var_inv_d_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 558 | 7 | 4 | block4 | Block4：Wo 操作数 encode 噪声 (`wo_sf`) | SF=22 | True | 22 | 来自 static_skeletons：encode（操作数噪声） |  |
| 559 | 7 | 4 | block4 | Block4：softmax mask 后 rescale 噪声 (`softmax_out_mask_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 560 | 7 | 4 | block4 | Block4：V mask 后 rescale 噪声 (`v_mask_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 561 | 7 | 4 | block4 | Block4：softmax*V 乘法后 rescale 噪声 (`softmax_v_matmul_rescale_sf`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 562 | 7 | 4 | block4 | Block4：softmax*V mask 后 rescale 噪声 (`softmax_v_mask_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 563 | 7 | 4 | block4 | Block4：At*Wo 后 rescale 噪声 (`wo_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 564 | 7 | 4 | block4 | Block4：LN 均值分支 rescale 噪声 (`ln_mean_rescale_sf`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 565 | 7 | 4 | block4 | Block4：(X-u)^2 后 rescale 噪声 (`ln_square_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 566 | 7 | 4 | block4 | Block4：LN 方差分支 rescale 噪声 (`ln_var_rescale_sf`) | SF=28 | True | 28 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 567 | 7 | 4 | block4 | Block4：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 568 | 7 | 5 | block5_n1 | Block5：FFN LN inv_std fresh 噪声 (`inv_std_fresh_sf`) | SF=30 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 569 | 7 | 5 | block5_n1 | Block5：FFN LN x_mean/X_centered fresh 噪声 (`x_centered_fresh_sf`) | SF=30 | True | 30 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 570 | 7 | 5 | block5_n1 | Block5：FFN LN gamma 操作数 encode 噪声 (`gamma_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 571 | 7 | 5 | block5_n1 | Block5：Wffn1 操作数 encode 噪声 (`wffn1_sf`) | SF=22 | True | 22 | 来自 static_skeletons：encode（操作数噪声） |  |
| 572 | 7 | 5 | block5_n1 | Block5：GELU 系数操作数 encode 噪声 (`gelu_coeff_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 573 | 7 | 5 | block5_n1 | Block5：(X-u)/std 后 rescale 噪声 (`normalize_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 574 | 7 | 5 | block5_n1 | Block5：乘 gamma 后 rescale 噪声 (`gamma_rescale_sf`) | SF=30 | True | 30 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 575 | 7 | 5 | block5_n1 | Block5：Wffn1*X 后 rescale 噪声 (`wffn1_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 576 | 7 | 5 | block5_n1 | Block5：GELU x^2 后 rescale 噪声 (`gelu_power_rescale_sf_0`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this power-rescale slot |
| 577 | 7 | 5 | block5_n1 | Block5：GELU x^3 后 rescale 噪声 (`gelu_power_rescale_sf_1`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this power-rescale slot |
| 578 | 7 | 5 | block5_n1 | Block5：GELU x^4 后 rescale 噪声 (`gelu_power_rescale_sf_2`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this power-rescale slot |
| 579 | 7 | 5 | block5_n1 | Block5：GELU degree=1 系数乘法后 rescale 噪声 (`gelu_coeff_mul_rescale_sf_0`) | SF=30 | True | 30 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 580 | 7 | 5 | block5_n1 | Block5：GELU degree=2 系数乘法后 rescale 噪声 (`gelu_coeff_mul_rescale_sf_1`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this coefficient-rescale slot |
| 581 | 7 | 5 | block5_n1 | Block5：GELU degree=3 占位系数 rescale 噪声 (`gelu_coeff_mul_rescale_sf_2`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this coefficient-rescale slot |
| 582 | 7 | 5 | block5_n1 | Block5：GELU degree=4 系数乘法后 rescale 噪声 (`gelu_coeff_mul_rescale_sf_3`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this coefficient-rescale slot |
| 583 | 7 | 5 | block5_n1 | Block5：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 584 | 8 | 1 | block1_mrpc | Block1：GELU 输出 fresh 噪声 (`gelu_out_sf`) | SF=30 | True | 30 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 585 | 8 | 1 | block1_mrpc | Block1：Wffn2 操作数 encode 噪声 (`wffn2_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 586 | 8 | 1 | block1_mrpc | Block1：第一个 1/d 操作数 encode 噪声 (`mean_inv_d_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 587 | 8 | 1 | block1_mrpc | Block1：第二个 1/d 操作数 encode 噪声 (`var_inv_d_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 588 | 8 | 1 | block1_mrpc | Block1：Wffn2*X 后结果 rescale 噪声 (`wffn2_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 589 | 8 | 1 | block1_mrpc | Block1：第一个 1/d 乘法后结果 rescale 噪声 (`mean_rescale_sf`) | SF=34 | True | 34 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 590 | 8 | 1 | block1_mrpc | Block1：(X-u)^2 后结果 rescale 噪声 (`square_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 591 | 8 | 1 | block1_mrpc | Block1：第二个 1/d 乘法后结果 rescale 噪声 (`var_rescale_sf`) | SF=34 | True | 34 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 592 | 8 | 1 | block1_mrpc | Block1：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 593 | 8 | 2 | block2_mrpc | Block2：inv_std/std 相关输入 fresh 噪声 (`inv_std_fresh_sf`) | SF=31 | True | 31 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 594 | 8 | 2 | block2_mrpc | Block2：X_centered 输入 fresh 噪声 (`x_centered_fresh_sf`) | SF=30 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 595 | 8 | 2 | block2_mrpc | Block2：LayerNorm gamma 操作数 encode 噪声 (`gamma_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 596 | 8 | 2 | block2_mrpc | Block2：Wq 操作数 encode 噪声（RO ctpt_wq_wk 共享） (`wq_sf`) | SF=22 | True | 22 | 来自 static_skeletons：encode（操作数噪声） |  |
| 597 | 8 | 2 | block2_mrpc | Block2：Wk 操作数 encode 噪声（RO ctpt_wq_wk 共享） (`wk_sf`) | SF=22 | True | 22 | 来自 static_skeletons：encode（操作数噪声） |  |
| 598 | 8 | 2 | block2_mrpc | Block2：Wv 操作数 encode 噪声 (`wv_sf`) | SF=22 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 599 | 8 | 2 | block2_mrpc | Block2：KT 第一个 mask 操作数 encode 噪声 (`kt_mask1_sf`) | SF=15 | True | 15 | 来自 static_skeletons：encode（操作数噪声） |  |
| 600 | 8 | 2 | block2_mrpc | Block2：KT 第二个 mask 操作数 encode 噪声 (`kt_mask2_sf`) | SF=15 | True | 15 | 来自 static_skeletons：encode（操作数噪声） |  |
| 601 | 8 | 2 | block2_mrpc | Block2：Q 第一个 mask 操作数 encode 噪声 (`q_mask1_sf`) | SF=15 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 602 | 8 | 2 | block2_mrpc | Block2：Q 第二个 mask 操作数 encode 噪声 (`q_mask2_sf`) | SF=15 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 603 | 8 | 2 | block2_mrpc | Block2：QKT merge mask 操作数 encode 噪声 (`qkt_merge_mask_sf`) | SF=15 | True | 15 | 来自 static_skeletons：encode（操作数噪声） |  |
| 604 | 8 | 2 | block2_mrpc | Block2：(X-u)/std 归一化后 rescale 噪声 (`normalize_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 605 | 8 | 2 | block2_mrpc | Block2：乘 gamma 后 rescale 噪声 (`gamma_rescale_sf`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 606 | 8 | 2 | block2_mrpc | Block2：WkX 后 rescale 噪声 (`wk_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 607 | 8 | 2 | block2_mrpc | Block2：WqX 后 rescale 噪声 (`wq_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 608 | 8 | 2 | block2_mrpc | Block2：WvX 后 rescale 噪声 (`wv_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 609 | 8 | 2 | block2_mrpc | Block2：KT 第一个 mask 后 rescale 噪声 (`kt_mask1_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 610 | 8 | 2 | block2_mrpc | Block2：KT 第二个 mask 后 rescale 噪声 (`kt_mask2_rescale_sf`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 611 | 8 | 2 | block2_mrpc | Block2：Q 第一个 mask 后 rescale 噪声 (`q_mask1_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 612 | 8 | 2 | block2_mrpc | Block2：Q 第二个 mask 后 rescale 噪声 (`q_mask2_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 613 | 8 | 2 | block2_mrpc | Block2：Q*KT 后 rescale 噪声 (`qkt_matmul_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 614 | 8 | 2 | block2_mrpc | Block2：QKT merge mask 后 rescale 噪声 (`qkt_merge_mask_rescale_sf`) | SF=28 | True | 28 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 615 | 8 | 2 | block2_mrpc | Block2：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 616 | 8 | 3 | block3_exp_n5 | Block3：softmax 输入 X fresh 噪声 (`x_fresh_sf`) | SF=28 | True | 28 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 617 | 8 | 3 | block3_exp_n5 | Block3：1/(2n) 操作数 encode 噪声 (`inv_2n_sf`) | SF=15 | True | 15 | 来自 static_skeletons：encode（操作数噪声） |  |
| 618 | 8 | 3 | block3_exp_n5 | Block3：X*(1/2n) 后 rescale 噪声 (`x_inv_2n_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 619 | 8 | 3 | block3_exp_n5 | Block3：softmax 平方链第 1 个 rescale 噪声 (`square_rescale_sf_0`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 620 | 8 | 3 | block3_exp_n5 | Block3：softmax 平方链第 2 个 rescale 噪声 (`square_rescale_sf_1`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 621 | 8 | 3 | block3_exp_n5 | Block3：softmax 平方链第 3 个 rescale 噪声 (`square_rescale_sf_2`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 622 | 8 | 3 | block3_exp_n5 | Block3：softmax 平方链第 4 个 rescale 噪声 (`square_rescale_sf_3`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 623 | 8 | 3 | block3_exp_n5 | Block3：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 624 | 8 | 4 | block4 | Block4：softmax 输出 fresh 噪声 (`softmax_out_fresh_sf`) | SF=35 | True | 35 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 625 | 8 | 4 | block4 | Block4：V 输入 fresh 噪声 (`v_fresh_sf`) | SF=30 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 626 | 8 | 4 | block4 | Block4：softmax mask 操作数 encode 噪声 (`softmax_out_mask_sf`) | SF=14 | True | 14 | 来自 static_skeletons：encode（操作数噪声） |  |
| 627 | 8 | 4 | block4 | Block4：V mask 操作数 encode 噪声 (`v_mask_sf`) | SF=14 | True | 14 | 来自 static_skeletons：encode（操作数噪声） |  |
| 628 | 8 | 4 | block4 | Block4：softmax*V mask 操作数 encode 噪声 (`softmax_v_mask_sf`) | SF=14 | True | 14 | 来自 static_skeletons：encode（操作数噪声） |  |
| 629 | 8 | 4 | block4 | Block4：attention LN 第一个 1/d encode 噪声 (`ln_mean_inv_d_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 630 | 8 | 4 | block4 | Block4：attention LN 第二个 1/d encode 噪声 (`ln_var_inv_d_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 631 | 8 | 4 | block4 | Block4：Wo 操作数 encode 噪声 (`wo_sf`) | SF=22 | True | 22 | 来自 static_skeletons：encode（操作数噪声） |  |
| 632 | 8 | 4 | block4 | Block4：softmax mask 后 rescale 噪声 (`softmax_out_mask_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 633 | 8 | 4 | block4 | Block4：V mask 后 rescale 噪声 (`v_mask_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 634 | 8 | 4 | block4 | Block4：softmax*V 乘法后 rescale 噪声 (`softmax_v_matmul_rescale_sf`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 635 | 8 | 4 | block4 | Block4：softmax*V mask 后 rescale 噪声 (`softmax_v_mask_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 636 | 8 | 4 | block4 | Block4：At*Wo 后 rescale 噪声 (`wo_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 637 | 8 | 4 | block4 | Block4：LN 均值分支 rescale 噪声 (`ln_mean_rescale_sf`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 638 | 8 | 4 | block4 | Block4：(X-u)^2 后 rescale 噪声 (`ln_square_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 639 | 8 | 4 | block4 | Block4：LN 方差分支 rescale 噪声 (`ln_var_rescale_sf`) | SF=28 | True | 28 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 640 | 8 | 4 | block4 | Block4：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 641 | 8 | 5 | block5_n1 | Block5：FFN LN inv_std fresh 噪声 (`inv_std_fresh_sf`) | SF=30 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 642 | 8 | 5 | block5_n1 | Block5：FFN LN x_mean/X_centered fresh 噪声 (`x_centered_fresh_sf`) | SF=30 | True | 30 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 643 | 8 | 5 | block5_n1 | Block5：FFN LN gamma 操作数 encode 噪声 (`gamma_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 644 | 8 | 5 | block5_n1 | Block5：Wffn1 操作数 encode 噪声 (`wffn1_sf`) | SF=22 | True | 22 | 来自 static_skeletons：encode（操作数噪声） |  |
| 645 | 8 | 5 | block5_n1 | Block5：GELU 系数操作数 encode 噪声 (`gelu_coeff_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 646 | 8 | 5 | block5_n1 | Block5：(X-u)/std 后 rescale 噪声 (`normalize_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 647 | 8 | 5 | block5_n1 | Block5：乘 gamma 后 rescale 噪声 (`gamma_rescale_sf`) | SF=30 | True | 30 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 648 | 8 | 5 | block5_n1 | Block5：Wffn1*X 后 rescale 噪声 (`wffn1_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 649 | 8 | 5 | block5_n1 | Block5：GELU x^2 后 rescale 噪声 (`gelu_power_rescale_sf_0`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this power-rescale slot |
| 650 | 8 | 5 | block5_n1 | Block5：GELU x^3 后 rescale 噪声 (`gelu_power_rescale_sf_1`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this power-rescale slot |
| 651 | 8 | 5 | block5_n1 | Block5：GELU x^4 后 rescale 噪声 (`gelu_power_rescale_sf_2`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this power-rescale slot |
| 652 | 8 | 5 | block5_n1 | Block5：GELU degree=1 系数乘法后 rescale 噪声 (`gelu_coeff_mul_rescale_sf_0`) | SF=30 | True | 30 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 653 | 8 | 5 | block5_n1 | Block5：GELU degree=2 系数乘法后 rescale 噪声 (`gelu_coeff_mul_rescale_sf_1`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this coefficient-rescale slot |
| 654 | 8 | 5 | block5_n1 | Block5：GELU degree=3 占位系数 rescale 噪声 (`gelu_coeff_mul_rescale_sf_2`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this coefficient-rescale slot |
| 655 | 8 | 5 | block5_n1 | Block5：GELU degree=4 系数乘法后 rescale 噪声 (`gelu_coeff_mul_rescale_sf_3`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this coefficient-rescale slot |
| 656 | 8 | 5 | block5_n1 | Block5：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 657 | 9 | 1 | block1_mrpc | Block1：GELU 输出 fresh 噪声 (`gelu_out_sf`) | SF=30 | True | 30 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 658 | 9 | 1 | block1_mrpc | Block1：Wffn2 操作数 encode 噪声 (`wffn2_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 659 | 9 | 1 | block1_mrpc | Block1：第一个 1/d 操作数 encode 噪声 (`mean_inv_d_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 660 | 9 | 1 | block1_mrpc | Block1：第二个 1/d 操作数 encode 噪声 (`var_inv_d_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 661 | 9 | 1 | block1_mrpc | Block1：Wffn2*X 后结果 rescale 噪声 (`wffn2_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 662 | 9 | 1 | block1_mrpc | Block1：第一个 1/d 乘法后结果 rescale 噪声 (`mean_rescale_sf`) | SF=34 | True | 34 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 663 | 9 | 1 | block1_mrpc | Block1：(X-u)^2 后结果 rescale 噪声 (`square_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 664 | 9 | 1 | block1_mrpc | Block1：第二个 1/d 乘法后结果 rescale 噪声 (`var_rescale_sf`) | SF=34 | True | 34 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 665 | 9 | 1 | block1_mrpc | Block1：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 666 | 9 | 2 | block2_mrpc | Block2：inv_std/std 相关输入 fresh 噪声 (`inv_std_fresh_sf`) | SF=31 | True | 31 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 667 | 9 | 2 | block2_mrpc | Block2：X_centered 输入 fresh 噪声 (`x_centered_fresh_sf`) | SF=30 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 668 | 9 | 2 | block2_mrpc | Block2：LayerNorm gamma 操作数 encode 噪声 (`gamma_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 669 | 9 | 2 | block2_mrpc | Block2：Wq 操作数 encode 噪声（RO ctpt_wq_wk 共享） (`wq_sf`) | SF=22 | True | 22 | 来自 static_skeletons：encode（操作数噪声） |  |
| 670 | 9 | 2 | block2_mrpc | Block2：Wk 操作数 encode 噪声（RO ctpt_wq_wk 共享） (`wk_sf`) | SF=22 | True | 22 | 来自 static_skeletons：encode（操作数噪声） |  |
| 671 | 9 | 2 | block2_mrpc | Block2：Wv 操作数 encode 噪声 (`wv_sf`) | SF=22 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 672 | 9 | 2 | block2_mrpc | Block2：KT 第一个 mask 操作数 encode 噪声 (`kt_mask1_sf`) | SF=15 | True | 15 | 来自 static_skeletons：encode（操作数噪声） |  |
| 673 | 9 | 2 | block2_mrpc | Block2：KT 第二个 mask 操作数 encode 噪声 (`kt_mask2_sf`) | SF=15 | True | 15 | 来自 static_skeletons：encode（操作数噪声） |  |
| 674 | 9 | 2 | block2_mrpc | Block2：Q 第一个 mask 操作数 encode 噪声 (`q_mask1_sf`) | SF=15 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 675 | 9 | 2 | block2_mrpc | Block2：Q 第二个 mask 操作数 encode 噪声 (`q_mask2_sf`) | SF=15 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 676 | 9 | 2 | block2_mrpc | Block2：QKT merge mask 操作数 encode 噪声 (`qkt_merge_mask_sf`) | SF=15 | True | 15 | 来自 static_skeletons：encode（操作数噪声） |  |
| 677 | 9 | 2 | block2_mrpc | Block2：(X-u)/std 归一化后 rescale 噪声 (`normalize_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 678 | 9 | 2 | block2_mrpc | Block2：乘 gamma 后 rescale 噪声 (`gamma_rescale_sf`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 679 | 9 | 2 | block2_mrpc | Block2：WkX 后 rescale 噪声 (`wk_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 680 | 9 | 2 | block2_mrpc | Block2：WqX 后 rescale 噪声 (`wq_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 681 | 9 | 2 | block2_mrpc | Block2：WvX 后 rescale 噪声 (`wv_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 682 | 9 | 2 | block2_mrpc | Block2：KT 第一个 mask 后 rescale 噪声 (`kt_mask1_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 683 | 9 | 2 | block2_mrpc | Block2：KT 第二个 mask 后 rescale 噪声 (`kt_mask2_rescale_sf`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 684 | 9 | 2 | block2_mrpc | Block2：Q 第一个 mask 后 rescale 噪声 (`q_mask1_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 685 | 9 | 2 | block2_mrpc | Block2：Q 第二个 mask 后 rescale 噪声 (`q_mask2_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 686 | 9 | 2 | block2_mrpc | Block2：Q*KT 后 rescale 噪声 (`qkt_matmul_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 687 | 9 | 2 | block2_mrpc | Block2：QKT merge mask 后 rescale 噪声 (`qkt_merge_mask_rescale_sf`) | SF=28 | True | 28 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 688 | 9 | 2 | block2_mrpc | Block2：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 689 | 9 | 3 | block3_exp_n5 | Block3：softmax 输入 X fresh 噪声 (`x_fresh_sf`) | SF=28 | True | 28 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 690 | 9 | 3 | block3_exp_n5 | Block3：1/(2n) 操作数 encode 噪声 (`inv_2n_sf`) | SF=15 | True | 15 | 来自 static_skeletons：encode（操作数噪声） |  |
| 691 | 9 | 3 | block3_exp_n5 | Block3：X*(1/2n) 后 rescale 噪声 (`x_inv_2n_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 692 | 9 | 3 | block3_exp_n5 | Block3：softmax 平方链第 1 个 rescale 噪声 (`square_rescale_sf_0`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 693 | 9 | 3 | block3_exp_n5 | Block3：softmax 平方链第 2 个 rescale 噪声 (`square_rescale_sf_1`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 694 | 9 | 3 | block3_exp_n5 | Block3：softmax 平方链第 3 个 rescale 噪声 (`square_rescale_sf_2`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 695 | 9 | 3 | block3_exp_n5 | Block3：softmax 平方链第 4 个 rescale 噪声 (`square_rescale_sf_3`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 696 | 9 | 3 | block3_exp_n5 | Block3：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 697 | 9 | 4 | block4 | Block4：softmax 输出 fresh 噪声 (`softmax_out_fresh_sf`) | SF=35 | True | 35 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 698 | 9 | 4 | block4 | Block4：V 输入 fresh 噪声 (`v_fresh_sf`) | SF=30 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 699 | 9 | 4 | block4 | Block4：softmax mask 操作数 encode 噪声 (`softmax_out_mask_sf`) | SF=14 | True | 14 | 来自 static_skeletons：encode（操作数噪声） |  |
| 700 | 9 | 4 | block4 | Block4：V mask 操作数 encode 噪声 (`v_mask_sf`) | SF=14 | True | 14 | 来自 static_skeletons：encode（操作数噪声） |  |
| 701 | 9 | 4 | block4 | Block4：softmax*V mask 操作数 encode 噪声 (`softmax_v_mask_sf`) | SF=14 | True | 14 | 来自 static_skeletons：encode（操作数噪声） |  |
| 702 | 9 | 4 | block4 | Block4：attention LN 第一个 1/d encode 噪声 (`ln_mean_inv_d_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 703 | 9 | 4 | block4 | Block4：attention LN 第二个 1/d encode 噪声 (`ln_var_inv_d_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 704 | 9 | 4 | block4 | Block4：Wo 操作数 encode 噪声 (`wo_sf`) | SF=22 | True | 22 | 来自 static_skeletons：encode（操作数噪声） |  |
| 705 | 9 | 4 | block4 | Block4：softmax mask 后 rescale 噪声 (`softmax_out_mask_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 706 | 9 | 4 | block4 | Block4：V mask 后 rescale 噪声 (`v_mask_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 707 | 9 | 4 | block4 | Block4：softmax*V 乘法后 rescale 噪声 (`softmax_v_matmul_rescale_sf`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 708 | 9 | 4 | block4 | Block4：softmax*V mask 后 rescale 噪声 (`softmax_v_mask_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 709 | 9 | 4 | block4 | Block4：At*Wo 后 rescale 噪声 (`wo_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 710 | 9 | 4 | block4 | Block4：LN 均值分支 rescale 噪声 (`ln_mean_rescale_sf`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 711 | 9 | 4 | block4 | Block4：(X-u)^2 后 rescale 噪声 (`ln_square_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 712 | 9 | 4 | block4 | Block4：LN 方差分支 rescale 噪声 (`ln_var_rescale_sf`) | SF=28 | True | 28 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 713 | 9 | 4 | block4 | Block4：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 714 | 9 | 5 | block5_n1 | Block5：FFN LN inv_std fresh 噪声 (`inv_std_fresh_sf`) | SF=30 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 715 | 9 | 5 | block5_n1 | Block5：FFN LN x_mean/X_centered fresh 噪声 (`x_centered_fresh_sf`) | SF=30 | True | 30 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 716 | 9 | 5 | block5_n1 | Block5：FFN LN gamma 操作数 encode 噪声 (`gamma_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 717 | 9 | 5 | block5_n1 | Block5：Wffn1 操作数 encode 噪声 (`wffn1_sf`) | SF=22 | True | 22 | 来自 static_skeletons：encode（操作数噪声） |  |
| 718 | 9 | 5 | block5_n1 | Block5：GELU 系数操作数 encode 噪声 (`gelu_coeff_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 719 | 9 | 5 | block5_n1 | Block5：(X-u)/std 后 rescale 噪声 (`normalize_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 720 | 9 | 5 | block5_n1 | Block5：乘 gamma 后 rescale 噪声 (`gamma_rescale_sf`) | SF=30 | True | 30 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 721 | 9 | 5 | block5_n1 | Block5：Wffn1*X 后 rescale 噪声 (`wffn1_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 722 | 9 | 5 | block5_n1 | Block5：GELU x^2 后 rescale 噪声 (`gelu_power_rescale_sf_0`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this power-rescale slot |
| 723 | 9 | 5 | block5_n1 | Block5：GELU x^3 后 rescale 噪声 (`gelu_power_rescale_sf_1`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this power-rescale slot |
| 724 | 9 | 5 | block5_n1 | Block5：GELU x^4 后 rescale 噪声 (`gelu_power_rescale_sf_2`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this power-rescale slot |
| 725 | 9 | 5 | block5_n1 | Block5：GELU degree=1 系数乘法后 rescale 噪声 (`gelu_coeff_mul_rescale_sf_0`) | SF=30 | True | 30 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 726 | 9 | 5 | block5_n1 | Block5：GELU degree=2 系数乘法后 rescale 噪声 (`gelu_coeff_mul_rescale_sf_1`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this coefficient-rescale slot |
| 727 | 9 | 5 | block5_n1 | Block5：GELU degree=3 占位系数 rescale 噪声 (`gelu_coeff_mul_rescale_sf_2`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this coefficient-rescale slot |
| 728 | 9 | 5 | block5_n1 | Block5：GELU degree=4 系数乘法后 rescale 噪声 (`gelu_coeff_mul_rescale_sf_3`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this coefficient-rescale slot |
| 729 | 9 | 5 | block5_n1 | Block5：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 730 | 10 | 1 | block1_mrpc | Block1：GELU 输出 fresh 噪声 (`gelu_out_sf`) | SF=30 | True | 30 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 731 | 10 | 1 | block1_mrpc | Block1：Wffn2 操作数 encode 噪声 (`wffn2_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 732 | 10 | 1 | block1_mrpc | Block1：第一个 1/d 操作数 encode 噪声 (`mean_inv_d_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 733 | 10 | 1 | block1_mrpc | Block1：第二个 1/d 操作数 encode 噪声 (`var_inv_d_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 734 | 10 | 1 | block1_mrpc | Block1：Wffn2*X 后结果 rescale 噪声 (`wffn2_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 735 | 10 | 1 | block1_mrpc | Block1：第一个 1/d 乘法后结果 rescale 噪声 (`mean_rescale_sf`) | SF=34 | True | 34 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 736 | 10 | 1 | block1_mrpc | Block1：(X-u)^2 后结果 rescale 噪声 (`square_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 737 | 10 | 1 | block1_mrpc | Block1：第二个 1/d 乘法后结果 rescale 噪声 (`var_rescale_sf`) | SF=34 | True | 34 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 738 | 10 | 1 | block1_mrpc | Block1：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 739 | 10 | 2 | block2_mrpc | Block2：inv_std/std 相关输入 fresh 噪声 (`inv_std_fresh_sf`) | SF=31 | True | 31 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 740 | 10 | 2 | block2_mrpc | Block2：X_centered 输入 fresh 噪声 (`x_centered_fresh_sf`) | SF=30 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 741 | 10 | 2 | block2_mrpc | Block2：LayerNorm gamma 操作数 encode 噪声 (`gamma_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 742 | 10 | 2 | block2_mrpc | Block2：Wq 操作数 encode 噪声（RO ctpt_wq_wk 共享） (`wq_sf`) | SF=22 | True | 22 | 来自 static_skeletons：encode（操作数噪声） |  |
| 743 | 10 | 2 | block2_mrpc | Block2：Wk 操作数 encode 噪声（RO ctpt_wq_wk 共享） (`wk_sf`) | SF=22 | True | 22 | 来自 static_skeletons：encode（操作数噪声） |  |
| 744 | 10 | 2 | block2_mrpc | Block2：Wv 操作数 encode 噪声 (`wv_sf`) | SF=22 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 745 | 10 | 2 | block2_mrpc | Block2：KT 第一个 mask 操作数 encode 噪声 (`kt_mask1_sf`) | SF=15 | True | 15 | 来自 static_skeletons：encode（操作数噪声） |  |
| 746 | 10 | 2 | block2_mrpc | Block2：KT 第二个 mask 操作数 encode 噪声 (`kt_mask2_sf`) | SF=15 | True | 15 | 来自 static_skeletons：encode（操作数噪声） |  |
| 747 | 10 | 2 | block2_mrpc | Block2：Q 第一个 mask 操作数 encode 噪声 (`q_mask1_sf`) | SF=15 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 748 | 10 | 2 | block2_mrpc | Block2：Q 第二个 mask 操作数 encode 噪声 (`q_mask2_sf`) | SF=15 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 749 | 10 | 2 | block2_mrpc | Block2：QKT merge mask 操作数 encode 噪声 (`qkt_merge_mask_sf`) | SF=15 | True | 15 | 来自 static_skeletons：encode（操作数噪声） |  |
| 750 | 10 | 2 | block2_mrpc | Block2：(X-u)/std 归一化后 rescale 噪声 (`normalize_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 751 | 10 | 2 | block2_mrpc | Block2：乘 gamma 后 rescale 噪声 (`gamma_rescale_sf`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 752 | 10 | 2 | block2_mrpc | Block2：WkX 后 rescale 噪声 (`wk_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 753 | 10 | 2 | block2_mrpc | Block2：WqX 后 rescale 噪声 (`wq_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 754 | 10 | 2 | block2_mrpc | Block2：WvX 后 rescale 噪声 (`wv_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 755 | 10 | 2 | block2_mrpc | Block2：KT 第一个 mask 后 rescale 噪声 (`kt_mask1_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 756 | 10 | 2 | block2_mrpc | Block2：KT 第二个 mask 后 rescale 噪声 (`kt_mask2_rescale_sf`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 757 | 10 | 2 | block2_mrpc | Block2：Q 第一个 mask 后 rescale 噪声 (`q_mask1_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 758 | 10 | 2 | block2_mrpc | Block2：Q 第二个 mask 后 rescale 噪声 (`q_mask2_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 759 | 10 | 2 | block2_mrpc | Block2：Q*KT 后 rescale 噪声 (`qkt_matmul_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 760 | 10 | 2 | block2_mrpc | Block2：QKT merge mask 后 rescale 噪声 (`qkt_merge_mask_rescale_sf`) | SF=28 | True | 28 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 761 | 10 | 2 | block2_mrpc | Block2：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 762 | 10 | 3 | block3_exp_n6 | Block3：softmax 输入 X fresh 噪声 (`x_fresh_sf`) | SF=28 | True | 28 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 763 | 10 | 3 | block3_exp_n6 | Block3：1/(2n) 操作数 encode 噪声 (`inv_2n_sf`) | SF=15 | True | 15 | 来自 static_skeletons：encode（操作数噪声） |  |
| 764 | 10 | 3 | block3_exp_n6 | Block3：X*(1/2n) 后 rescale 噪声 (`x_inv_2n_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 765 | 10 | 3 | block3_exp_n6 | Block3：softmax 平方链第 1 个 rescale 噪声 (`square_rescale_sf_0`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 766 | 10 | 3 | block3_exp_n6 | Block3：softmax 平方链第 2 个 rescale 噪声 (`square_rescale_sf_1`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 767 | 10 | 3 | block3_exp_n6 | Block3：softmax 平方链第 3 个 rescale 噪声 (`square_rescale_sf_2`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 768 | 10 | 3 | block3_exp_n6 | Block3：softmax 平方链第 4 个 rescale 噪声 (`square_rescale_sf_3`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 769 | 10 | 3 | block3_exp_n6 | Block3：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 770 | 10 | 4 | block4 | Block4：softmax 输出 fresh 噪声 (`softmax_out_fresh_sf`) | SF=35 | True | 35 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 771 | 10 | 4 | block4 | Block4：V 输入 fresh 噪声 (`v_fresh_sf`) | SF=30 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 772 | 10 | 4 | block4 | Block4：softmax mask 操作数 encode 噪声 (`softmax_out_mask_sf`) | SF=14 | True | 14 | 来自 static_skeletons：encode（操作数噪声） |  |
| 773 | 10 | 4 | block4 | Block4：V mask 操作数 encode 噪声 (`v_mask_sf`) | SF=14 | True | 14 | 来自 static_skeletons：encode（操作数噪声） |  |
| 774 | 10 | 4 | block4 | Block4：softmax*V mask 操作数 encode 噪声 (`softmax_v_mask_sf`) | SF=14 | True | 14 | 来自 static_skeletons：encode（操作数噪声） |  |
| 775 | 10 | 4 | block4 | Block4：attention LN 第一个 1/d encode 噪声 (`ln_mean_inv_d_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 776 | 10 | 4 | block4 | Block4：attention LN 第二个 1/d encode 噪声 (`ln_var_inv_d_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 777 | 10 | 4 | block4 | Block4：Wo 操作数 encode 噪声 (`wo_sf`) | SF=22 | True | 22 | 来自 static_skeletons：encode（操作数噪声） |  |
| 778 | 10 | 4 | block4 | Block4：softmax mask 后 rescale 噪声 (`softmax_out_mask_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 779 | 10 | 4 | block4 | Block4：V mask 后 rescale 噪声 (`v_mask_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 780 | 10 | 4 | block4 | Block4：softmax*V 乘法后 rescale 噪声 (`softmax_v_matmul_rescale_sf`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 781 | 10 | 4 | block4 | Block4：softmax*V mask 后 rescale 噪声 (`softmax_v_mask_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 782 | 10 | 4 | block4 | Block4：At*Wo 后 rescale 噪声 (`wo_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 783 | 10 | 4 | block4 | Block4：LN 均值分支 rescale 噪声 (`ln_mean_rescale_sf`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 784 | 10 | 4 | block4 | Block4：(X-u)^2 后 rescale 噪声 (`ln_square_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 785 | 10 | 4 | block4 | Block4：LN 方差分支 rescale 噪声 (`ln_var_rescale_sf`) | SF=28 | True | 28 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 786 | 10 | 4 | block4 | Block4：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 787 | 10 | 5 | block5_n1 | Block5：FFN LN inv_std fresh 噪声 (`inv_std_fresh_sf`) | SF=30 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 788 | 10 | 5 | block5_n1 | Block5：FFN LN x_mean/X_centered fresh 噪声 (`x_centered_fresh_sf`) | SF=30 | True | 30 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 789 | 10 | 5 | block5_n1 | Block5：FFN LN gamma 操作数 encode 噪声 (`gamma_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 790 | 10 | 5 | block5_n1 | Block5：Wffn1 操作数 encode 噪声 (`wffn1_sf`) | SF=22 | True | 22 | 来自 static_skeletons：encode（操作数噪声） |  |
| 791 | 10 | 5 | block5_n1 | Block5：GELU 系数操作数 encode 噪声 (`gelu_coeff_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 792 | 10 | 5 | block5_n1 | Block5：(X-u)/std 后 rescale 噪声 (`normalize_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 793 | 10 | 5 | block5_n1 | Block5：乘 gamma 后 rescale 噪声 (`gamma_rescale_sf`) | SF=30 | True | 30 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 794 | 10 | 5 | block5_n1 | Block5：Wffn1*X 后 rescale 噪声 (`wffn1_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 795 | 10 | 5 | block5_n1 | Block5：GELU x^2 后 rescale 噪声 (`gelu_power_rescale_sf_0`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this power-rescale slot |
| 796 | 10 | 5 | block5_n1 | Block5：GELU x^3 后 rescale 噪声 (`gelu_power_rescale_sf_1`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this power-rescale slot |
| 797 | 10 | 5 | block5_n1 | Block5：GELU x^4 后 rescale 噪声 (`gelu_power_rescale_sf_2`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this power-rescale slot |
| 798 | 10 | 5 | block5_n1 | Block5：GELU degree=1 系数乘法后 rescale 噪声 (`gelu_coeff_mul_rescale_sf_0`) | SF=30 | True | 30 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 799 | 10 | 5 | block5_n1 | Block5：GELU degree=2 系数乘法后 rescale 噪声 (`gelu_coeff_mul_rescale_sf_1`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this coefficient-rescale slot |
| 800 | 10 | 5 | block5_n1 | Block5：GELU degree=3 占位系数 rescale 噪声 (`gelu_coeff_mul_rescale_sf_2`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this coefficient-rescale slot |
| 801 | 10 | 5 | block5_n1 | Block5：GELU degree=4 系数乘法后 rescale 噪声 (`gelu_coeff_mul_rescale_sf_3`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this coefficient-rescale slot |
| 802 | 10 | 5 | block5_n1 | Block5：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 803 | 11 | 1 | block1_mrpc | Block1：GELU 输出 fresh 噪声 (`gelu_out_sf`) | SF=30 | True | 30 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 804 | 11 | 1 | block1_mrpc | Block1：Wffn2 操作数 encode 噪声 (`wffn2_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 805 | 11 | 1 | block1_mrpc | Block1：第一个 1/d 操作数 encode 噪声 (`mean_inv_d_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 806 | 11 | 1 | block1_mrpc | Block1：第二个 1/d 操作数 encode 噪声 (`var_inv_d_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 807 | 11 | 1 | block1_mrpc | Block1：Wffn2*X 后结果 rescale 噪声 (`wffn2_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 808 | 11 | 1 | block1_mrpc | Block1：第一个 1/d 乘法后结果 rescale 噪声 (`mean_rescale_sf`) | SF=34 | True | 34 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 809 | 11 | 1 | block1_mrpc | Block1：(X-u)^2 后结果 rescale 噪声 (`square_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 810 | 11 | 1 | block1_mrpc | Block1：第二个 1/d 乘法后结果 rescale 噪声 (`var_rescale_sf`) | SF=34 | True | 34 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 811 | 11 | 1 | block1_mrpc | Block1：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 812 | 11 | 2 | block2_mrpc | Block2：inv_std/std 相关输入 fresh 噪声 (`inv_std_fresh_sf`) | SF=31 | True | 31 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 813 | 11 | 2 | block2_mrpc | Block2：X_centered 输入 fresh 噪声 (`x_centered_fresh_sf`) | SF=30 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 814 | 11 | 2 | block2_mrpc | Block2：LayerNorm gamma 操作数 encode 噪声 (`gamma_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 815 | 11 | 2 | block2_mrpc | Block2：Wq 操作数 encode 噪声（RO ctpt_wq_wk 共享） (`wq_sf`) | SF=22 | True | 22 | 来自 static_skeletons：encode（操作数噪声） |  |
| 816 | 11 | 2 | block2_mrpc | Block2：Wk 操作数 encode 噪声（RO ctpt_wq_wk 共享） (`wk_sf`) | SF=22 | True | 22 | 来自 static_skeletons：encode（操作数噪声） |  |
| 817 | 11 | 2 | block2_mrpc | Block2：Wv 操作数 encode 噪声 (`wv_sf`) | SF=22 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 818 | 11 | 2 | block2_mrpc | Block2：KT 第一个 mask 操作数 encode 噪声 (`kt_mask1_sf`) | SF=15 | True | 15 | 来自 static_skeletons：encode（操作数噪声） |  |
| 819 | 11 | 2 | block2_mrpc | Block2：KT 第二个 mask 操作数 encode 噪声 (`kt_mask2_sf`) | SF=15 | True | 15 | 来自 static_skeletons：encode（操作数噪声） |  |
| 820 | 11 | 2 | block2_mrpc | Block2：Q 第一个 mask 操作数 encode 噪声 (`q_mask1_sf`) | SF=15 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 821 | 11 | 2 | block2_mrpc | Block2：Q 第二个 mask 操作数 encode 噪声 (`q_mask2_sf`) | SF=15 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 822 | 11 | 2 | block2_mrpc | Block2：QKT merge mask 操作数 encode 噪声 (`qkt_merge_mask_sf`) | SF=15 | True | 15 | 来自 static_skeletons：encode（操作数噪声） |  |
| 823 | 11 | 2 | block2_mrpc | Block2：(X-u)/std 归一化后 rescale 噪声 (`normalize_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 824 | 11 | 2 | block2_mrpc | Block2：乘 gamma 后 rescale 噪声 (`gamma_rescale_sf`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 825 | 11 | 2 | block2_mrpc | Block2：WkX 后 rescale 噪声 (`wk_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 826 | 11 | 2 | block2_mrpc | Block2：WqX 后 rescale 噪声 (`wq_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 827 | 11 | 2 | block2_mrpc | Block2：WvX 后 rescale 噪声 (`wv_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 828 | 11 | 2 | block2_mrpc | Block2：KT 第一个 mask 后 rescale 噪声 (`kt_mask1_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 829 | 11 | 2 | block2_mrpc | Block2：KT 第二个 mask 后 rescale 噪声 (`kt_mask2_rescale_sf`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 830 | 11 | 2 | block2_mrpc | Block2：Q 第一个 mask 后 rescale 噪声 (`q_mask1_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 831 | 11 | 2 | block2_mrpc | Block2：Q 第二个 mask 后 rescale 噪声 (`q_mask2_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 832 | 11 | 2 | block2_mrpc | Block2：Q*KT 后 rescale 噪声 (`qkt_matmul_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 833 | 11 | 2 | block2_mrpc | Block2：QKT merge mask 后 rescale 噪声 (`qkt_merge_mask_rescale_sf`) | SF=28 | True | 28 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 834 | 11 | 2 | block2_mrpc | Block2：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 835 | 11 | 3 | block3_exp_n2 | Block3：softmax 输入 X fresh 噪声 (`x_fresh_sf`) | SF=27 | True | 27 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 836 | 11 | 3 | block3_exp_n2 | Block3：1/(2n) 操作数 encode 噪声 (`inv_2n_sf`) | SF=16 | True | 16 | 来自 static_skeletons：encode（操作数噪声） |  |
| 837 | 11 | 3 | block3_exp_n2 | Block3：X*(1/2n) 后 rescale 噪声 (`x_inv_2n_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 838 | 11 | 3 | block3_exp_n2 | Block3：softmax 平方链第 1 个 rescale 噪声 (`square_rescale_sf_0`) | SF=34 | True | 34 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 839 | 11 | 3 | block3_exp_n2 | Block3：softmax 平方链第 2 个 rescale 噪声 (`square_rescale_sf_1`) | SF=34 | True | 34 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 840 | 11 | 3 | block3_exp_n2 | Block3：softmax 平方链第 3 个 rescale 噪声 (`square_rescale_sf_2`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | softmax degree 2 does not use this square-rescale slot |
| 841 | 11 | 3 | block3_exp_n2 | Block3：softmax 平方链第 4 个 rescale 噪声 (`square_rescale_sf_3`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | softmax degree 2 does not use this square-rescale slot |
| 842 | 11 | 3 | block3_exp_n2 | Block3：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 843 | 11 | 4 | block4 | Block4：softmax 输出 fresh 噪声 (`softmax_out_fresh_sf`) | SF=35 | True | 35 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 844 | 11 | 4 | block4 | Block4：V 输入 fresh 噪声 (`v_fresh_sf`) | SF=30 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 845 | 11 | 4 | block4 | Block4：softmax mask 操作数 encode 噪声 (`softmax_out_mask_sf`) | SF=14 | True | 14 | 来自 static_skeletons：encode（操作数噪声） |  |
| 846 | 11 | 4 | block4 | Block4：V mask 操作数 encode 噪声 (`v_mask_sf`) | SF=14 | True | 14 | 来自 static_skeletons：encode（操作数噪声） |  |
| 847 | 11 | 4 | block4 | Block4：softmax*V mask 操作数 encode 噪声 (`softmax_v_mask_sf`) | SF=14 | True | 14 | 来自 static_skeletons：encode（操作数噪声） |  |
| 848 | 11 | 4 | block4 | Block4：attention LN 第一个 1/d encode 噪声 (`ln_mean_inv_d_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 849 | 11 | 4 | block4 | Block4：attention LN 第二个 1/d encode 噪声 (`ln_var_inv_d_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 850 | 11 | 4 | block4 | Block4：Wo 操作数 encode 噪声 (`wo_sf`) | SF=22 | True | 22 | 来自 static_skeletons：encode（操作数噪声） |  |
| 851 | 11 | 4 | block4 | Block4：softmax mask 后 rescale 噪声 (`softmax_out_mask_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 852 | 11 | 4 | block4 | Block4：V mask 后 rescale 噪声 (`v_mask_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 853 | 11 | 4 | block4 | Block4：softmax*V 乘法后 rescale 噪声 (`softmax_v_matmul_rescale_sf`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 854 | 11 | 4 | block4 | Block4：softmax*V mask 后 rescale 噪声 (`softmax_v_mask_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 855 | 11 | 4 | block4 | Block4：At*Wo 后 rescale 噪声 (`wo_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 856 | 11 | 4 | block4 | Block4：LN 均值分支 rescale 噪声 (`ln_mean_rescale_sf`) | SF=31 | True | 31 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 857 | 11 | 4 | block4 | Block4：(X-u)^2 后 rescale 噪声 (`ln_square_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 858 | 11 | 4 | block4 | Block4：LN 方差分支 rescale 噪声 (`ln_var_rescale_sf`) | SF=28 | True | 28 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 859 | 11 | 4 | block4 | Block4：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 860 | 11 | 5 | block5_n1 | Block5：FFN LN inv_std fresh 噪声 (`inv_std_fresh_sf`) | SF=30 | True |  | archive 未显式给出：保留动作空间默认 max |  |
| 861 | 11 | 5 | block5_n1 | Block5：FFN LN x_mean/X_centered fresh 噪声 (`x_centered_fresh_sf`) | SF=30 | True | 30 | 来自 static_skeletons：fresh（固定 fresh 噪声） |  |
| 862 | 11 | 5 | block5_n1 | Block5：FFN LN gamma 操作数 encode 噪声 (`gamma_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 863 | 11 | 5 | block5_n1 | Block5：Wffn1 操作数 encode 噪声 (`wffn1_sf`) | SF=22 | True | 22 | 来自 static_skeletons：encode（操作数噪声） |  |
| 864 | 11 | 5 | block5_n1 | Block5：GELU 系数操作数 encode 噪声 (`gelu_coeff_sf`) | SF=20 | True | 20 | 来自 static_skeletons：encode（操作数噪声） |  |
| 865 | 11 | 5 | block5_n1 | Block5：(X-u)/std 后 rescale 噪声 (`normalize_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 866 | 11 | 5 | block5_n1 | Block5：乘 gamma 后 rescale 噪声 (`gamma_rescale_sf`) | SF=30 | True | 30 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 867 | 11 | 5 | block5_n1 | Block5：Wffn1*X 后 rescale 噪声 (`wffn1_rescale_sf`) | OFF（不加 rescale 噪声） | True |  | static_skeletons 未出现 sf_post/drop：rescale=OFF |  |
| 868 | 11 | 5 | block5_n1 | Block5：GELU x^2 后 rescale 噪声 (`gelu_power_rescale_sf_0`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this power-rescale slot |
| 869 | 11 | 5 | block5_n1 | Block5：GELU x^3 后 rescale 噪声 (`gelu_power_rescale_sf_1`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this power-rescale slot |
| 870 | 11 | 5 | block5_n1 | Block5：GELU x^4 后 rescale 噪声 (`gelu_power_rescale_sf_2`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this power-rescale slot |
| 871 | 11 | 5 | block5_n1 | Block5：GELU degree=1 系数乘法后 rescale 噪声 (`gelu_coeff_mul_rescale_sf_0`) | SF=30 | True | 30 | 来自 static_skeletons：rescale（sf_post/drop） |  |
| 872 | 11 | 5 | block5_n1 | Block5：GELU degree=2 系数乘法后 rescale 噪声 (`gelu_coeff_mul_rescale_sf_1`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this coefficient-rescale slot |
| 873 | 11 | 5 | block5_n1 | Block5：GELU degree=3 占位系数 rescale 噪声 (`gelu_coeff_mul_rescale_sf_2`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this coefficient-rescale slot |
| 874 | 11 | 5 | block5_n1 | Block5：GELU degree=4 系数乘法后 rescale 噪声 (`gelu_coeff_mul_rescale_sf_3`) | OFF（不加 rescale 噪声） | False |  | static_skeletons 未出现 sf_post/drop：rescale=OFF | GELU degree 1 does not use this coefficient-rescale slot |
| 875 | 11 | 5 | block5_n1 | Block5：输出截断 k (`output_truncation_k`) | k=13 | True |  | RO baseline 不返回 K：默认最大 k |  |
| 876 | 0 | first_input | first_input_L0 | 首层输入 fresh 噪声（当前语义废弃，不安装） (`first_input_sf`) | SF=30 | False |  | 无效槽：first_input 已废弃 | first_input fresh noise deprecated; the first HE config is treated as lossless. Slot kept for action-vector backward compatibility. |
