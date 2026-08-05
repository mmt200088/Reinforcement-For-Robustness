# BLB Stage-2 RL Live Summary

- Run: `s1t0.001_s2t0.001_s2st2.0__bertbase_mrpc_stage2_k3_4gpu_stage1best20260624_5c222da6_20260804`
- Profile: `mrpc`
- Phase: max_episodes_reached
- Updated at: 2026-08-05T01:02:34.056130
- Elapsed seconds: 7255.664000
- Episode: 60000 / 60000 (100.00%)
- PPO updates: 501

## Reward

- Last reward: +1.437926
- Recent reward mean: +0.928500
- Best reward: +1.604782
- Best episode: n/a
- Last priority: 3
- Last invalid: False

## Last Terminal Metrics


## Last PPO Update

- `policy_loss`: -0.002936398610472679
- `value_loss`: 0.2005738914012909
- `entropy`: 0.42836302518844604
- `clip_fraction`: 0.001953125
- `window_mean_return`: 0.97662925144943
- `window_mean_invalid`: 0.0
- `approx_kl`: 5.03960472997278e-06
- `lr`: 5e-05
- `ent_coef`: 0.0

## Key Artifacts

- Status JSON: `blb_stage2_status.json`
- Live summary: `blb_stage2_live_summary.md`
- Episodes: `diagnostics/episodes.jsonl`
- PPO updates: `diagnostics/ppo_updates.jsonl`
- Diagnostics summary: `diagnostics/diagnostics_summary.md`
- Details batches: `details/`
- Training curve: `blb_stage2_training_curve.png`
- Entropy curve: `blb_stage2_entropy_curve.png`
- Final report: `blb_stage2_report.md`
