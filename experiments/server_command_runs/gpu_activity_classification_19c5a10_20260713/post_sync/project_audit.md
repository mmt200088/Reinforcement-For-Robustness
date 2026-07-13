# Project Optimization Audit

Root: `/hy-tmp/rfr_runtime_optimization`

## Summary

- Flow stages: 6
- Expected files: 30
- Present files: 30
- Missing files: 0

## Flow Stages

### launcher: Launcher, presets, and server bridge
- Present files: 5
- Missing files: 0
- `llama_7B_LayerImportance.sh`: present
- `presets/mrpc-blb-stage2-rl.conf`: present
- `Paean/run_final_eval.sh`: present
- `SERVER_COMMAND.md`: present
- `scripts/server_resource_snapshot.py`: present
- Optimization surfaces: GPU flag forwarding, server source sync, run manifest, strict resource gates

### stage1: Stage-1 plaintext RL and validation
- Present files: 5
- Missing files: 0
- `layer_importance_evaluator.py`: present
- `stage1_rl/parallel_runner.py`: present
- `stage1_rl/eval_cache.py`: present
- `function_handler.py`: present
- `scripts/stage1_parallel_report.py`: present
- Optimization surfaces: validation_full forward reuse, multi-GPU rollout collection, deterministic eval cache, hot-path report decoupling

### stage2: Stage-2 BLB RL and reward probes
- Present files: 6
- Missing files: 0
- `blb_stage2_rl/parallel_runner.py`: present
- `blb_stage2_rl/probe_runner.py`: present
- `blb_stage2_rl/sequential_runner.py`: present
- `scripts/stage2_ngpu_ab_compare.py`: present
- `scripts/gpu_utilization_report.py`: present
- `scripts/stage2_reward_probe_scaling_report.py`: present
- Optimization surfaces: episode-parallel GPU workers, reward-probe device balance, replan/probe timing, JSONL write overhead

### rescale: Rescale optimizer and fusion maps
- Present files: 4
- Missing files: 0
- `Rescale_optimizer/rescale_optimizer/replan_interface.py`: present
- `Rescale_optimizer/rescale_optimizer/replan.py`: present
- `scripts/blb_build_fusion_count_map.py`: present
- `blb_stage2_rl/fusion_count_map.py`: present
- Optimization surfaces: ReplanSession reuse, graph/baseline cache, streaming fusion-map build, CPU worker scheduling

### paean: Paean final evaluation
- Present files: 5
- Missing files: 0
- `Paean/run_final_eval.py`: present
- `Paean/config.py`: present
- `Paean/action_grid.py`: present
- `Paean/blb_action_eval.py`: present
- `final_evaluation_module.py`: present
- Optimization surfaces: model/tokenizer reuse, action-grid batching, independent-config scheduling, report/render decoupling

### artifacts: Structured data, reports, and sync
- Present files: 5
- Missing files: 0
- `rl_data_points.py`: present
- `scripts/verify_stage2_persistent_outputs.py`: present
- `scripts/optimization_evidence_bundle.py`: present
- `tools/paper_figures.py`: present
- `experiments/index.md`: present
- Optimization surfaces: complete JSON/JSONL mirrors, compact hot-path writes, post-run report rendering, artifact/source commit linkage

## Artifact Evidence

- Roots scanned: 3
- episodes_jsonl: 13
- html_reports: 117
- npz_curves: 18
- nvidia_smi_csv: 20
- ppo_updates_jsonl: 19
- status_json: 27
- Missing evidence: none

## Next Steps

- Run this audit before and after performance work.
- Use server_resource_snapshot.py before expensive server runs.
- Use stage1_parallel_report.py for Stage-1 rollout/cache timing evidence.
- Use gpu_utilization_report.py for run-level GPU evidence.
- Use optimization_evidence_bundle.py to package server evidence before promotion.
- Use stage2_ngpu_ab_compare.py before promoting Stage-2 GPU defaults.
- Keep report rendering off the training hot path when possible.
